import sys
import unittest
from pathlib import Path


PAYLOAD = Path(__file__).resolve().parent.parent
if str(PAYLOAD) not in sys.path:
    sys.path.insert(0, str(PAYLOAD))


class ReleaseHygieneTests(unittest.TestCase):
    def test_machine_local_authorization_policy_is_never_tracked(self):
        ignored_paths = (PAYLOAD / ".gitignore").read_text(encoding="utf-8").splitlines()

        self.assertIn("telegram_roles.json", ignored_paths)

    def test_pytest_collects_only_the_reviewed_test_suite(self):
        config = (PAYLOAD / "pytest.ini").read_text(encoding="utf-8")

        self.assertIn("testpaths = tests", config)
