import ast
import contextlib
import io
from pathlib import Path
import unittest
import tempfile
from unittest import mock


class AuditReleaseTests(unittest.TestCase):
    def test_binary_text_never_reaches_semantic_model(self):
        from cerebro_core.mayordomo_review import semantic_classify
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'corrupt.txt'
            path.write_bytes(b'\xff' * 8192)
            with mock.patch('cerebro_core.mayordomo_review.local_http_open') as request:
                category, _ = semantic_classify(path)
            self.assertEqual(category, 'Documentos_PDF')
            request.assert_not_called()

    def test_doctor_does_not_repair_without_explicit_flag(self):
        from cerebro_cli import __main__ as cli
        report = {'status': 'degraded', 'ai_inference_state': 'failed',
                  'composer_online': False, 'jarvis_online': False}
        with mock.patch.object(cli, 'build_doctor_report', return_value=report), \
                mock.patch('auto_doctor.repair_local_services') as repair, \
                mock.patch.object(cli, 'exit_code_for_report', return_value=0), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(['doctor', '--json'])
        repair.assert_not_called()

    def test_release_includes_local_python_import_closure(self):
        from cerebro_release.installer import REVIEWED_FILES
        root = Path(__file__).resolve().parents[1]
        allowed = set(REVIEWED_FILES)
        for rel in allowed:
            if not rel.endswith('.py'):
                continue
            for node in ast.walk(ast.parse((root / rel).read_bytes())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                           [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    dependency = module.replace('.', '/') + '.py'
                    if (root / dependency).is_file():
                        self.assertIn(dependency, allowed, f'{rel} imports {dependency}')

    def test_published_launcher_uses_only_the_portable_cli(self):
        from cerebro_release.packager import PORTABLE_LAUNCHER
        self.assertIn('-m cerebro_cli', PORTABLE_LAUNCHER)
        self.assertNotIn('/home/example', PORTABLE_LAUNCHER)
        self.assertNotIn('WEBUI_AUTH', PORTABLE_LAUNCHER)
