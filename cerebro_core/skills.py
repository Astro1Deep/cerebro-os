"""Registro declarativo de skills; la lectura nunca ejecuta una skill."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = ("name", "version", "permissions", "entrypoint")


def list_skill_manifests(skills_dir: Path) -> list[dict[str, object]]:
    """Return valid manifests in a disabled state, sorted by skill name."""
    if not skills_dir.is_dir():
        return []

    skills: list[dict[str, object]] = []
    for path in sorted(skills_dir.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or any(field not in manifest for field in REQUIRED_FIELDS):
            continue
        if not isinstance(manifest["name"], str) or not manifest["name"].strip():
            continue
        if not isinstance(manifest["version"], str) or not isinstance(manifest["entrypoint"], str):
            continue
        permissions = manifest["permissions"]
        if not isinstance(permissions, list) or not all(isinstance(item, str) for item in permissions):
            continue
        skills.append(
            {
                "name": manifest["name"],
                "version": manifest["version"],
                "permissions": permissions,
                "entrypoint": manifest["entrypoint"],
                "state": "disabled",
            }
        )
    return skills
