"""Inspección segura del adaptador opcional de Hermes."""

from __future__ import annotations

import shutil
from pathlib import Path


def inspect_hermes(hermes_home: Path) -> dict[str, object]:
    """Report installation metadata without parsing configuration or secrets."""
    config_files = [
        hermes_home / name
        for name in ("config.yaml", "config.yml", "config.json")
        if (hermes_home / name).is_file()
    ]
    config_present = bool(config_files)
    detected = hermes_home.is_dir() or shutil.which("hermes") is not None
    connector_state = "not_configured"
    connector_target = "http://127.0.0.1:8080/v1"
    for cfg_path in config_files:
        try:
            raw_content = cfg_path.read_text(encoding="utf-8", errors="ignore")
            content = "\n".join(line.split("#", 1)[0] for line in raw_content.splitlines())
            if "127.0.0.1:8081/v1" in content or "mlxstudio" in content:
                connector_state = "configured"
                connector_target = "http://127.0.0.1:8081/v1"
                break
            if "127.0.0.1:8080/v1" in content:
                connector_state = "configured"
                connector_target = "http://127.0.0.1:8080/v1"
                break
            if "127.0.0.1:11434/v1" in content:
                connector_state = "configured"
                connector_target = "http://127.0.0.1:11434/v1"
                break
        except OSError:
            pass
    return {
        "hermes": {
            "state": "detected" if detected else "not_detected",
            "binary_detected": shutil.which("hermes") is not None,
            "config_present": config_present,
            "credentials_read": False,
        },
        "connector": {
            "state": connector_state,
            "target": connector_target,
        },
    }

