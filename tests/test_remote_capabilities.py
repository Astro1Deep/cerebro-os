import os
from datetime import UTC, datetime, timedelta
import stat
import tempfile
import unittest
from pathlib import Path


class RemoteCapabilityGrantTests(unittest.TestCase):
    def test_missing_policy_means_no_sensitive_capability_is_enabled(self):
        from cerebro_core.remote_capabilities import load_remote_capability_grants

        with tempfile.TemporaryDirectory() as tmp:
            grants = load_remote_capability_grants(Path(tmp) / "missing.json")

        self.assertFalse(grants.is_granted("screen"))
        self.assertFalse(grants.is_granted("camera"))

    def test_grants_expire_and_are_saved_owner_only(self):
        from cerebro_core.remote_capabilities import (
            RemoteCapabilityGrants,
            grant_capability,
            load_remote_capability_grants,
            save_remote_capability_grants,
        )

        now = datetime(2026, 9, 2, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telegram_capabilities.json"
            grants = grant_capability(RemoteCapabilityGrants({}), "screen", 30, now=now)
            save_remote_capability_grants(path, grants)
            restored = load_remote_capability_grants(path)

            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertTrue(restored.is_granted("screen", now=now + timedelta(minutes=29)))
            self.assertFalse(restored.is_granted("screen", now=now + timedelta(minutes=30)))
            self.assertFalse(restored.is_granted("camera", now=now))

    def test_invalid_grant_duration_and_unknown_capability_fail_closed(self):
        from cerebro_core.remote_capabilities import CapabilityPolicyError, RemoteCapabilityGrants, grant_capability

        with self.assertRaises(CapabilityPolicyError):
            grant_capability(RemoteCapabilityGrants({}), "screen", 0)
        with self.assertRaises(CapabilityPolicyError):
            grant_capability(RemoteCapabilityGrants({}), "shell", 10)

    def test_black_camera_frame_is_not_accepted_as_a_capture(self):
        from cerebro_core.remote_capture import _is_meaningfully_visible_frame

        class Frame:
            def __init__(self, maximum, mean):
                self.maximum = maximum
                self.average = mean

            def max(self):
                return self.maximum

            def mean(self):
                return self.average

        self.assertFalse(_is_meaningfully_visible_frame(Frame(0, 0.0)))
        self.assertFalse(_is_meaningfully_visible_frame(Frame(11, 8.0)))
        self.assertFalse(_is_meaningfully_visible_frame(Frame(255, 1.4)))
        self.assertTrue(_is_meaningfully_visible_frame(Frame(255, 12.0)))
