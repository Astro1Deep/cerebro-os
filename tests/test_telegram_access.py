import os
import importlib.util
import asyncio
import json
from unittest import mock
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PAYLOAD = Path(__file__).resolve().parent.parent
if str(PAYLOAD) not in sys.path:
    sys.path.insert(0, str(PAYLOAD))


class TelegramAccessPolicyTests(unittest.TestCase):
    def test_telegram_error_handler_records_only_the_exception_type(self):
        from telegram_bridge import log_telegram_error

        context = SimpleNamespace(error=RuntimeError("private details must not be logged"))
        with mock.patch("telegram_bridge.logging.error") as logger:
            asyncio.run(log_telegram_error(None, context))

        logger.assert_called_once_with("Telegram handler failure: %s", "RuntimeError")

    def test_telegram_publishes_only_the_reviewed_remote_command_menu(self):
        from telegram_bridge import TELEGRAM_COMMANDS, publish_safe_command_menu

        class FakeBot:
            def __init__(self):
                self.commands = None

            async def set_my_commands(self, commands):
                self.commands = commands

        bot = FakeBot()
        asyncio.run(publish_safe_command_menu(SimpleNamespace(bot=bot)))

        self.assertEqual(
            [command.command for command in TELEGRAM_COMMANDS],
            [
                "start",
                "status",
                "doctor",
                "ask",
                "remember",
                "memories",
                "research",
                "learn",
                "sources",
                "composer",
                "commercial",
                "flipper",
                "screen",
                "cam",
            ],
        )
        self.assertEqual(bot.commands, TELEGRAM_COMMANDS)

    def test_telegram_allows_a_cold_local_model_load_before_reporting_timeout(self):
        from telegram_bridge import MODEL_REQUEST_TIMEOUT_SECONDS

        self.assertGreaterEqual(MODEL_REQUEST_TIMEOUT_SECONDS, 180)

    def test_access_policy_module_exists_before_the_bot_uses_it(self):
        self.assertIsNotNone(importlib.util.find_spec("cerebro_core.access_control"))

    def test_policy_is_fail_closed_and_limits_roles_to_safe_operations(self):
        from cerebro_core.access_control import RemoteAccessPolicy

        policy = RemoteAccessPolicy(owner_id=101, operators={202})

        self.assertTrue(policy.allows(101, "ask"))
        self.assertTrue(policy.allows(101, "doctor"))
        self.assertTrue(policy.allows(101, "remember"))
        self.assertTrue(policy.allows(101, "screen"))
        self.assertTrue(policy.allows(101, "camera"))
        self.assertTrue(policy.allows(101, "research"))
        self.assertTrue(policy.allows(101, "composer"))
        self.assertTrue(policy.allows(101, "flipper"))
        self.assertTrue(policy.allows(202, "status"))
        self.assertFalse(policy.allows(202, "remember"))
        self.assertFalse(policy.allows(202, "screen"))
        self.assertFalse(policy.allows(202, "research"))
        self.assertFalse(policy.allows(202, "manage_admins"))
        self.assertFalse(policy.allows(303, "ask"))
        self.assertFalse(policy.allows(101, "shell"))

    def test_policy_file_is_validated_and_written_owner_only(self):
        from cerebro_core.access_control import RemoteAccessPolicy, load_remote_access_policy, save_remote_access_policy

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telegram_roles.json"
            expected = RemoteAccessPolicy(owner_id=101, operators=frozenset({202}), root_alias="Astro One")

            save_remote_access_policy(path, expected)

            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(load_remote_access_policy(path), expected)

    def test_invalid_policy_never_grants_access(self):
        from cerebro_core.access_control import PolicyConfigurationError, load_remote_access_policy

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "telegram_roles.json"
            path.write_text(json.dumps({"version": 1, "owner_id": 0, "operators": []}), encoding="utf-8")

            with self.assertRaises(PolicyConfigurationError):
                load_remote_access_policy(path)

    def test_legacy_administrator_requires_an_explicit_local_migration(self):
        from cerebro_core.access_control import (
            load_remote_access_policy,
            migrate_legacy_owner,
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            legacy = base / "admin.txt"
            target = base / "telegram_roles.json"
            legacy.write_text("101\n", encoding="utf-8")

            migrated = migrate_legacy_owner(legacy, target, root_alias="Oráculo")

            self.assertEqual(migrated.owner_id, 101)
            self.assertEqual(migrated.root_alias, "Oráculo")
            self.assertEqual(load_remote_access_policy(target), migrated)

    def test_telegram_bridge_contains_no_shell_execution_path(self):
        bridge_source = (PAYLOAD / "telegram_bridge.py").read_text(encoding="utf-8")

        self.assertNotIn("create_subprocess_shell", bridge_source)
        self.assertNotIn("def run_shell", bridge_source)
        self.assertNotIn('CommandHandler("bash"', bridge_source)
        self.assertIn('CommandHandler("remember"', bridge_source)

    def test_screen_is_never_captured_without_a_local_expiring_grant(self):
        from cerebro_core.access_control import RemoteAccessPolicy
        from telegram_bridge import screen_cmd

        class FakeMessage:
            def __init__(self):
                self.replies = []

            async def reply_text(self, text, **_kwargs):
                self.replies.append(text)

        message = FakeMessage()
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=101, type="private"),
            effective_user=SimpleNamespace(id=101),
            effective_message=message,
        )
        context = SimpleNamespace(
            application=SimpleNamespace(bot_data={"remote_access_policy": RemoteAccessPolicy(101, frozenset())}),
            bot=SimpleNamespace(),
        )
        with mock.patch("telegram_bridge.CAPABILITY_FILE", Path(tempfile.gettempdir()) / "missing-capabilities.json"), mock.patch(
            "telegram_bridge.capture_screen"
        ) as capture:
            asyncio.run(screen_cmd(update, context))

        capture.assert_not_called()
        self.assertIn("cerebro telegram grant", message.replies[0])

    def test_telegram_can_only_call_the_exact_local_mlx_endpoint(self):
        from telegram_bridge import validate_local_ai_url

        self.assertEqual(
            validate_local_ai_url("http://127.0.0.1:8080/v1/chat/completions"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            validate_local_ai_url("http://127.0.0.1:11434/v1/chat/completions"),
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        with self.assertRaises(ValueError):
            validate_local_ai_url("https://model.example/v1/chat/completions")
        with self.assertRaises(ValueError):
            validate_local_ai_url("http://127.0.0.1/v1/chat/completions")

    def test_shared_local_ai_endpoint_policy_accepts_only_loopback_engines(self):
        from cerebro_core.local_endpoint import configured_local_ai_endpoint, validate_local_ai_endpoint

        self.assertEqual(
            validate_local_ai_endpoint("http://127.0.0.1:8080/v1/chat/completions"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            validate_local_ai_endpoint("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/chat/completions",
        )
        for unsafe_url in (
            "https://model.example/v1/chat/completions",
            "http://127.0.0.1:11434/v1/chat/completions?forward=outside",
            "http://user:pass@127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:8080/v1/models",
        ):
            with self.assertRaises(ValueError):
                validate_local_ai_endpoint(unsafe_url)
        self.assertEqual(
            configured_local_ai_endpoint("https://model.example/v1/chat/completions"),
            "http://127.0.0.1:8081/v1/chat/completions",
        )

    def test_runtime_configuration_reads_only_declared_keys_without_shell_evaluation(self):
        from telegram_bridge import load_private_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.env"
            path.write_text(
                "TELEGRAM_TOKEN=test-token\n"
                "CEREBRO_AI_URL=http://127.0.0.1:8080/v1/chat/completions\n"
                "UNRELATED=$(unsafe)\n",
                encoding="utf-8",
            )
            path.chmod(0o600)

            self.assertEqual(
                load_private_runtime_config(path),
                {
                    "TELEGRAM_TOKEN": "test-token",
                    "CEREBRO_AI_URL": "http://127.0.0.1:8080/v1/chat/completions",
                },
            )

    def test_runtime_configuration_accepts_plain_quoted_values_without_evaluation(self):
        from telegram_bridge import load_private_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.env"
            path.write_text("TELEGRAM_TOKEN='quoted-token'\n", encoding="utf-8")
            path.chmod(0o600)

            self.assertEqual(load_private_runtime_config(path)["TELEGRAM_TOKEN"], "quoted-token")

    def test_runtime_configuration_uses_windows_acl_instead_of_posix_mode_bits(self):
        from telegram_bridge import load_private_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.env"
            path.write_text("TELEGRAM_TOKEN=test-token\n", encoding="utf-8")
            path.chmod(0o644)
            with mock.patch("telegram_bridge.REQUIRE_OWNER_ONLY_CONFIG_MODE", False):
                self.assertEqual(load_private_runtime_config(path)["TELEGRAM_TOKEN"], "test-token")

    def test_telegram_includes_the_configured_ollama_model(self):
        from telegram_bridge import ask_mlx

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def json(self, **_kwargs):
                return {"choices": [{"message": {"content": "Consejo local"}}]}

        class FakeSession:
            def __init__(self, **_kwargs):
                self.payloads = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, _url, *, json, allow_redirects):
                assert not allow_redirects
                self.payloads.append(json)
                return FakeResponse()

        session = FakeSession()
        with (
            mock.patch("telegram_bridge.aiohttp.ClientSession", return_value=session),
            mock.patch("telegram_bridge.build_chat_context", return_value="Contexto local verificado"),
        ):
            reply = asyncio.run(
                ask_mlx(
                    [{"role": "user", "content": "hola"}],
                    "http://127.0.0.1:11434/v1/chat/completions",
                    "llama3.2",
                )
            )

        self.assertEqual(reply, "Consejo local")
        self.assertEqual(session.payloads[0]["model"], "llama3.2")
        self.assertIn("Contexto local verificado", session.payloads[0]["messages"][1]["content"])

    def test_telegram_bounds_history_before_sending_it_to_the_local_model(self):
        from telegram_bridge import ask_mlx

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def json(self, **_kwargs):
                return {"choices": [{"message": {"content": "Respuesta local"}}]}

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, _url, *, json, allow_redirects):
                assert not allow_redirects
                self.payload = json
                return FakeResponse()

        session = FakeSession()
        current = "consulta-actual-" + "U" * 1_100
        history = [
            {"role": "user", "content": "historial-antiguo-" + "A" * 2_000},
            {"role": "assistant", "content": "respuesta-reciente-" + "B" * 600},
            {"role": "user", "content": current},
        ]
        with (
            mock.patch("telegram_bridge.aiohttp.ClientSession", return_value=session),
            mock.patch("telegram_bridge.build_chat_context", return_value="C" * 1_600),
        ):
            asyncio.run(ask_mlx(history, "http://127.0.0.1:8080/v1/chat/completions"))

        messages = session.payload["messages"]
        self.assertLessEqual(sum(len(message["content"]) for message in messages), 4_500)
        self.assertEqual(messages[-1]["content"], current)
        self.assertNotIn("historial-antiguo", "\n".join(message["content"] for message in messages))

