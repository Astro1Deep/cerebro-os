"""Non-destructive installer for the portable Cerebro beta payload."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final

from cerebro_core.capabilities import select_engines
from cerebro_core.local_endpoint import validate_local_ai_model


FILES: Final[tuple[str, ...]] = (
    "VERSION",
    "cerebro",
    "auto_doctor.py",
    "cerebro_runtime.py",
    "composer_server.py",
    "composer_ui.html",
    "composer_audio.js",
    "mlx-mayordomo.py",
    "package.json",
    "telegram_bridge.py",
    "web_jarvis.py",
    "requirements-beta.txt",
    "requirements-beta-constraints.txt",
    "requirements-beta-apple-silicon.txt",
    "README.md",
    "README_ES.md",
    "docs/GUIDE.md",
    "docs/GUIA.md",
    "docs/RELEASE_NOTES.md",
)
TREE_FILES: Final[tuple[str, ...]] = (
    "cerebro_cli/__init__.py",
    "cerebro_cli/__main__.py",
    "cerebro_core/__init__.py",
    "cerebro_core/access_control.py",
    "cerebro_core/branding.py",
    "cerebro_core/capabilities.py",
    "cerebro_core/chat_budget.py",
    "cerebro_core/conversation_context.py",
    "cerebro_core/hermes.py",
    "cerebro_core/local_endpoint.py",
    "cerebro_core/mayordomo_review.py",
    "cerebro_core/persistent_memory.py",
    "cerebro_core/memory_migration.py",
    "cerebro_core/profiles.py",
    "cerebro_core/skills.py",
    "cerebro_core/status.py",
    "cerebro_core/supervisor.py",
    "cerebro_core/system_context.py",
    "cerebro_core/web_knowledge.py",
    "cerebro_core/commercial_research.py",
    "cerebro_core/evaluation.py",
    "cerebro_core/flipper_device.py",
    "cerebro_core/flipper_import.py",
    "cerebro_core/m4_optimizations.py",
    "cerebro_core/beatos.py",
    "cerebro_core/remote_capabilities.py",
    "cerebro_core/remote_capture.py",
    "cerebro_core/vector_engine.py",
    "cerebro_release/__init__.py",
    "cerebro_release/installer.py",
    "cerebro_release/packager.py",
    "cerebro_release/patcher.py",
)
REVIEWED_FILES: Final[tuple[str, ...]] = (*FILES, *TREE_FILES)
DIRECTORIES: Final[tuple[str, ...]] = ("cerebro_cli", "cerebro_core", "cerebro_release")
CONFIG_NAME: Final[str] = "config.env"


def endpoint_for_engine(engine: str) -> str:
    port = 8080 if engine == "mlx" else 11434
    return f"http://127.0.0.1:{port}/v1/chat/completions"


def default_model_for_engine(engine: str) -> str | None:
    """Return a lightweight documented default only where the engine requires it."""

    return "llama3.2" if engine == "ollama" else None


def _candidate_interpreters(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = [explicit] if explicit is not None else [Path(sys.executable)]
    if explicit is None:
        base_executable = getattr(sys, "_base_executable", "")
        if base_executable:
            candidates.append(Path(base_executable))
        for name in ("python3", "python"):
            discovered = shutil.which(name)
            if discovered:
                candidates.append(Path(discovered))
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique and resolved.is_file():
            unique.append(resolved)
    return unique


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _create_verified_venv(target_dir: Path, explicit: Path | None) -> Path:
    final_venv = target_dir / "venv"
    if final_venv.exists():
        raise FileExistsError("Ya existe venv; no se sustituye automáticamente")
    errors: list[str] = []
    for interpreter in _candidate_interpreters(explicit):
        staging = target_dir / f".venv-candidate-{uuid.uuid4().hex}"
        try:
            subprocess.run([str(interpreter), "-m", "venv", str(staging)], check=True)
            venv_python = _venv_python(staging)
            subprocess.run(
                [str(venv_python), "-c", "import encodings, sys; assert sys.version_info >= (3, 11)"],
                check=True,
            )
            staging.replace(final_venv)
            return _venv_python(final_venv)
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{interpreter.name}: {error}")
            shutil.rmtree(staging, ignore_errors=True)
    detail = "; ".join(errors) if errors else "no se encontró un intérprete Python válido"
    raise RuntimeError(f"No se pudo crear un entorno Python verificable: {detail}")


def install_payload(
    *,
    source_dir: Path,
    target_dir: Path,
    system: str,
    machine: str,
    dry_run: bool = False,
    create_venv: bool = True,
    install_dependencies: bool = True,
    python_executable: Path | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    """Copy only the reviewed beta surface, preserving local configuration.

    The function never removes a target file.  It deliberately omits legacy
    dashboard, mobile bridge, file-moving agents and WebUI integrations.
    """

    source_dir = source_dir.resolve()
    target_dir = target_dir.expanduser().resolve()
    if source_dir == target_dir:
        raise ValueError("El origen y el destino de instalación deben ser distintos")
    missing = [name for name in REVIEWED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"El paquete beta está incompleto: {', '.join(missing)}")

    engine = str(select_engines(system=system, machine=machine)["primary"])
    selected_model = validate_local_ai_model(model_name) if model_name else default_model_for_engine(engine)
    result: dict[str, object] = {
        "target": str(target_dir),
        "engine": engine,
        "endpoint": endpoint_for_engine(engine),
        "model": selected_model,
        "copied_files": list(REVIEWED_FILES),
        "copied_directories": list(DIRECTORIES),
        "dependencies_installed": False,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    if target_dir.exists() and (not target_dir.is_dir() or any(target_dir.iterdir())):
        raise FileExistsError("Usa un directorio de instalación nuevo o vacío")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in REVIEWED_FILES:
        destination = target_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / name, destination)

    config_path = target_dir / CONFIG_NAME
    if not config_path.exists():
        config_lines = [
            "# Endpoint local OpenAI-compatible; nunca se acepta una URL externa.",
            f"CEREBRO_AI_URL={endpoint_for_engine(engine)}",
        ]
        if selected_model:
            config_lines.append(f"CEREBRO_AI_MODEL={selected_model}")
        config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
        if os.name != "nt":
            config_path.chmod(0o600)

    if create_venv:
        python = _create_verified_venv(target_dir, python_executable)
        if install_dependencies:
            requirements = target_dir / (
                "requirements-beta-apple-silicon.txt" if engine == "mlx" else "requirements-beta.txt"
            )
            subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
            result["dependencies_installed"] = True
    return result
