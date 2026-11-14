"""Fail-closed authorization primitives for remote Cerebro interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Final


_OWNER_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "ask",
        "doctor",
        "status",
        "remember",
        "memory",
        "screen",
        "camera",
        "research",
        "learn",
        "sources",
        "composer",
        "commercial",
        "flipper",
    }
)
_OPERATOR_ACTIONS: Final[frozenset[str]] = frozenset({"ask", "doctor", "status"})
_POLICY_VERSION: Final[int] = 1
_DEFAULT_ROOT_ALIAS: Final[str] = "Astro One"


class PolicyConfigurationError(ValueError):
    """The policy file is missing or invalid, so access must be denied."""


@dataclass(frozen=True)
class RemoteAccessPolicy:
    """Authorization policy that has no remote execution capability by design."""

    owner_id: int
    operators: frozenset[int]
    root_alias: str = _DEFAULT_ROOT_ALIAS

    def __post_init__(self) -> None:
        if not _is_valid_user_id(self.owner_id):
            raise PolicyConfigurationError("owner_id must be a positive Telegram user id")
        normalized_operators = frozenset(self.operators)
        if any(not _is_valid_user_id(user_id) for user_id in normalized_operators):
            raise PolicyConfigurationError("operators must contain positive Telegram user ids")
        if self.owner_id in normalized_operators:
            raise PolicyConfigurationError("owner cannot also be an operator")
        if not isinstance(self.root_alias, str):
            raise PolicyConfigurationError("root_alias must be text")
        alias = self.root_alias.strip()
        if not alias or len(alias) > 64 or any(not character.isprintable() for character in alias):
            raise PolicyConfigurationError("root_alias must be a short printable alias")
        object.__setattr__(self, "operators", normalized_operators)
        object.__setattr__(self, "root_alias", alias)

    def allows(self, user_id: int | None, action: str) -> bool:
        if not isinstance(user_id, int) or user_id <= 0:
            return False
        if user_id == self.owner_id:
            return action in _OWNER_ACTIONS
        if user_id in self.operators:
            return action in _OPERATOR_ACTIONS
        return False


def _is_valid_user_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def load_remote_access_policy(path: Path | str) -> RemoteAccessPolicy:
    """Load a local policy file; no missing or malformed file is ever tolerated."""

    policy_path = Path(path)
    try:
        if os.name != "nt" and policy_path.stat().st_mode & 0o077:
            raise PolicyConfigurationError("remote access policy must be owner-only")
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyConfigurationError("remote access policy is unavailable") from error
    if not isinstance(raw, dict) or raw.get("version") != _POLICY_VERSION:
        raise PolicyConfigurationError("remote access policy version is invalid")
    owner_id = raw.get("owner_id")
    operators = raw.get("operators", [])
    root_alias = raw.get("root_alias", _DEFAULT_ROOT_ALIAS)
    if not isinstance(operators, list):
        raise PolicyConfigurationError("operators must be a list")
    try:
        return RemoteAccessPolicy(owner_id=owner_id, operators=frozenset(operators), root_alias=root_alias)
    except (TypeError, PolicyConfigurationError) as error:
        raise PolicyConfigurationError("remote access policy contents are invalid") from error


def save_remote_access_policy(path: Path | str, policy: RemoteAccessPolicy) -> None:
    """Atomically persist one local policy with owner-only filesystem access."""

    policy_path = Path(path)
    if not policy_path.parent.is_dir():
        raise PolicyConfigurationError("remote access policy directory does not exist")
    payload = {
        "version": _POLICY_VERSION,
        "owner_id": policy.owner_id,
        "operators": sorted(policy.operators),
        "root_alias": policy.root_alias,
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=policy_path.parent, prefix=f".{policy_path.name}.", delete=False
        ) as temporary_file:
            temporary_name = temporary_file.name
            if os.name != "nt":
                os.fchmod(temporary_file.fileno(), 0o600)
            json.dump(payload, temporary_file, ensure_ascii=False, sort_keys=True, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, policy_path)
        if os.name != "nt":
            os.chmod(policy_path, 0o600)
    except OSError as error:
        raise PolicyConfigurationError("remote access policy could not be saved") from error
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def migrate_legacy_owner(
    legacy_admin_path: Path | str,
    policy_path: Path | str,
    *,
    root_alias: str = _DEFAULT_ROOT_ALIAS,
) -> RemoteAccessPolicy:
    """One-time, local-only migration from the historic single-admin file.

    This function is deliberately never called by a Telegram update handler.
    """

    legacy_path = Path(legacy_admin_path)
    destination = Path(policy_path)
    if destination.exists():
        raise PolicyConfigurationError("remote access policy already exists")
    try:
        owner_id = int(legacy_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise PolicyConfigurationError("legacy administrator file is invalid") from error
    policy = RemoteAccessPolicy(owner_id=owner_id, operators=frozenset(), root_alias=root_alias)
    save_remote_access_policy(destination, policy)
    return policy
