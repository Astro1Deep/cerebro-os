"""Locally-issued, expiring grants for sensitive Telegram actions.

Telegram identity alone is not an adequate reason to activate a camera or send
screen contents.  This module keeps those actions disabled until the Mac owner
enables a short-lived grant from the local CLI.  The bridge can only read a
grant; it can never create, extend, or revoke one from a Telegram update.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Final


CAPABILITY_POLICY_VERSION: Final[int] = 1
SENSITIVE_CAPABILITIES: Final[frozenset[str]] = frozenset({"camera", "screen"})
MINIMUM_GRANT_MINUTES: Final[int] = 1
MAXIMUM_GRANT_MINUTES: Final[int] = 24 * 60


class CapabilityPolicyError(ValueError):
    """A capability policy cannot safely be read or changed."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CapabilityPolicyError("la caducidad de la capacidad no es válida")
    try:
        timestamp = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CapabilityPolicyError("la caducidad de la capacidad no es válida") from error
    if timestamp.tzinfo is None:
        raise CapabilityPolicyError("la caducidad de la capacidad no es válida")
    return timestamp.astimezone(UTC)


def _require_capability(value: object) -> str:
    if not isinstance(value, str) or value not in SENSITIVE_CAPABILITIES:
        raise CapabilityPolicyError("capacidad remota no reconocida")
    return value


@dataclass(frozen=True)
class RemoteCapabilityGrants:
    """Validated expiry timestamps for owner-only sensitive operations."""

    expires_at: dict[str, datetime]

    def is_granted(self, capability: str, *, now: datetime | None = None) -> bool:
        normalized = _require_capability(capability)
        current = _utc_now() if now is None else now.astimezone(UTC)
        expiry = self.expires_at.get(normalized)
        return bool(expiry and expiry > current)

    def expiry_for(self, capability: str) -> datetime | None:
        return self.expires_at.get(_require_capability(capability))


def load_remote_capability_grants(path: Path | str) -> RemoteCapabilityGrants:
    """Read a locally-managed policy, treating absence or invalid data as no grant."""

    policy_path = Path(path)
    if not policy_path.exists():
        return RemoteCapabilityGrants(expires_at={})
    try:
        if os.name != "nt" and policy_path.stat().st_mode & 0o077:
            raise CapabilityPolicyError("la política de capacidades debe ser privada")
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapabilityPolicyError("la política de capacidades no está disponible") from error
    if not isinstance(raw, dict) or raw.get("version") != CAPABILITY_POLICY_VERSION:
        raise CapabilityPolicyError("la política de capacidades no es válida")
    grants = raw.get("grants", {})
    if not isinstance(grants, dict):
        raise CapabilityPolicyError("la política de capacidades no es válida")
    parsed: dict[str, datetime] = {}
    for capability, expires_at in grants.items():
        parsed[_require_capability(capability)] = _parse_utc_timestamp(expires_at)
    return RemoteCapabilityGrants(expires_at=parsed)


def save_remote_capability_grants(path: Path | str, grants: RemoteCapabilityGrants) -> None:
    """Persist an explicitly approved grant atomically with owner-only access."""

    policy_path = Path(path)
    if not policy_path.parent.is_dir():
        raise CapabilityPolicyError("el directorio de la política no existe")
    payload = {
        "version": CAPABILITY_POLICY_VERSION,
        "grants": {
            capability: _as_utc_timestamp(expiry)
            for capability, expiry in sorted(grants.expires_at.items())
        },
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
        raise CapabilityPolicyError("la política de capacidades no se pudo guardar") from error
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def grant_capability(
    grants: RemoteCapabilityGrants,
    capability: str,
    minutes: int,
    *,
    now: datetime | None = None,
) -> RemoteCapabilityGrants:
    """Return a bounded, expiring grant; callers must persist it locally."""

    normalized = _require_capability(capability)
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not MINIMUM_GRANT_MINUTES <= minutes <= MAXIMUM_GRANT_MINUTES:
        raise CapabilityPolicyError(
            f"la duración debe estar entre {MINIMUM_GRANT_MINUTES} y {MAXIMUM_GRANT_MINUTES} minutos"
        )
    current = _utc_now() if now is None else now.astimezone(UTC)
    updated = dict(grants.expires_at)
    updated[normalized] = current + timedelta(minutes=minutes)
    return RemoteCapabilityGrants(expires_at=updated)


def revoke_capability(grants: RemoteCapabilityGrants, capability: str) -> RemoteCapabilityGrants:
    """Return grants with one capability disabled immediately."""

    normalized = _require_capability(capability)
    updated = dict(grants.expires_at)
    updated.pop(normalized, None)
    return RemoteCapabilityGrants(expires_at=updated)
