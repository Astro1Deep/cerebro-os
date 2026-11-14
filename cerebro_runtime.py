#!/usr/bin/env python3
"""Safe local service launcher and runtime configuration for Cerebro beta."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Final

from cerebro_core.local_endpoint import (
    DEFAULT_LOCAL_AI_ENDPOINT, validate_local_ai_endpoint, validate_local_ai_model,
)


RUNTIME_CONFIG_NAME: Final[str] = "config.env"
RUNTIME_CONFIG_KEYS: Final[frozenset[str]] = frozenset({"CEREBRO_AI_URL", "CEREBRO_AI_MODEL"})
REQUIRE_OWNER_ONLY_CONFIG_MODE: Final[bool] = os.name != "nt"


def load_runtime_config(base_dir: Path) -> dict[str, str]:
    """Read the allowlisted, non-executable local runtime configuration."""

    path = base_dir / RUNTIME_CONFIG_NAME
    if not path.is_file():
        return {}
    if REQUIRE_OWNER_ONLY_CONFIG_MODE and path.stat().st_mode & 0o077:
        raise ValueError("config.env debe ser accesible solo por su propietario")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in RUNTIME_CONFIG_KEYS:
            continue
        values[key] = (
            validate_local_ai_endpoint(value.strip())
            if key == "CEREBRO_AI_URL"
            else validate_local_ai_model(value.strip())
        )
    return values


def configure_local_endpoint(base_dir: Path) -> str:
    """Set the validated endpoint before importing a service module."""

    config = load_runtime_config(base_dir)
    endpoint = validate_local_ai_endpoint(os.environ.get("CEREBRO_AI_URL") or config.get("CEREBRO_AI_URL") or DEFAULT_LOCAL_AI_ENDPOINT)
    os.environ["CEREBRO_AI_URL"] = endpoint
    model = os.environ.get("CEREBRO_AI_MODEL") or config.get("CEREBRO_AI_MODEL")
    if model:
        os.environ["CEREBRO_AI_MODEL"] = validate_local_ai_model(model)
    return endpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arrancador local seguro de Cerebro")
    parser.add_argument("service", choices=("composer", "jarvis", "telegram"))
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    base_dir = args.base_dir.resolve()
    if base_dir != Path(__file__).resolve().parent:
        parser.error("Ejecuta el lanzador desde la instalación que quieras iniciar")
    configure_local_endpoint(base_dir)

    if args.service == "composer":
        import composer_server
        import uvicorn

        uvicorn.run(composer_server.app, host="127.0.0.1", port=8082)
        return 0
    if args.service == "jarvis":
        import web_jarvis

        web_jarvis.app.run(host=web_jarvis.BIND_HOST, port=web_jarvis.PORT, threaded=True)
        return 0

    import telegram_bridge

    telegram_bridge.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
