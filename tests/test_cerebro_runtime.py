import sys
import tempfile
import unittest
from pathlib import Path


PAYLOAD = Path(__file__).resolve().parent.parent
if str(PAYLOAD) not in sys.path:
    sys.path.insert(0, str(PAYLOAD))


class CerebroRuntimeTests(unittest.TestCase):
    def test_runtime_config_reads_only_the_local_model_endpoint(self):
        from cerebro_runtime import load_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / "config.env"
            config.write_text(
                "CEREBRO_AI_URL=http://127.0.0.1:11434/v1/chat/completions\n"
                "CEREBRO_AI_MODEL=llama3.2\n"
                "PATH=$(unsafe)\n"
                "TELEGRAM_TOKEN=never-return-this\n",
                encoding="utf-8",
            )
            config.chmod(0o600)

            self.assertEqual(
                load_runtime_config(base),
                {
                    "CEREBRO_AI_URL": "http://127.0.0.1:11434/v1/chat/completions",
                    "CEREBRO_AI_MODEL": "llama3.2",
                },
            )

    def test_runtime_config_rejects_an_external_model_endpoint(self):
        from cerebro_runtime import load_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / "config.env"
            config.write_text(
                "CEREBRO_AI_URL=https://example.invalid/v1/chat/completions\n",
                encoding="utf-8",
            )
            config.chmod(0o600)

            with self.assertRaises(ValueError):
                load_runtime_config(base)

    def test_runtime_config_rejects_an_unsafe_model_identifier(self):
        from cerebro_runtime import load_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / "config.env"
            config.write_text("CEREBRO_AI_MODEL=$(unsafe)\n", encoding="utf-8")
            config.chmod(0o600)

            with self.assertRaises(ValueError):
                load_runtime_config(base)

    def test_status_reads_the_installed_runtime_config_without_shell_sourcing(self):
        from cerebro_core.status import build_status

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "config.env").write_text(
                "CEREBRO_AI_URL=http://127.0.0.1:11434/v1/chat/completions\n",
                encoding="utf-8",
            )
            (base / "config.env").chmod(0o600)
            report = build_status(base, probe_network=False)

        self.assertEqual(
            report["services"]["ai"]["endpoint"],
            "http://127.0.0.1:11434/v1/models",
        )

    def test_doctor_reads_the_installed_runtime_config_without_shell_sourcing(self):
        from auto_doctor import build_report
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ("composer_server.py", "composer_ui.html", "VERSION"):
                (base / name).write_text("test", encoding="utf-8")
            (base / "config.env").write_text(
                "CEREBRO_AI_URL=http://127.0.0.1:11434/v1/chat/completions\n",
                encoding="utf-8",
            )
            (base / "config.env").chmod(0o600)
            with mock.patch("auto_doctor.probe_json", return_value={}) as probe:
                build_report(base, probe_network=True, required_modules=())

        self.assertIn(mock.call("http://127.0.0.1:11434/v1/models"), probe.call_args_list)


if __name__ == "__main__":
    unittest.main()
