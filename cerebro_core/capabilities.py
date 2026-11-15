"""Política portable de selección de motores locales."""

from __future__ import annotations

import platform
import shutil

APPLE_SILICON_ARCHITECTURES = {"arm64", "aarch64"}


def select_engines(*, system: str, machine: str) -> dict[str, object]:
    """Return the portable engine policy without probing or installing anything."""
    normalized_system = system.strip().lower()
    normalized_machine = machine.strip().lower()
    if normalized_system == "darwin" and normalized_machine in APPLE_SILICON_ARCHITECTURES:
        return {"primary": "mlx", "fallbacks": ["ollama", "llama_cpp"]}
    return {"primary": "ollama", "fallbacks": ["llama_cpp"]}


def build_capability_report() -> dict[str, object]:
    """Describe the current host without starting a process or downloading a model."""
    system = platform.system()
    machine = platform.machine()
    binaries = {
        "mlx": shutil.which("mlx_lm.server") is not None,
        "llama_cpp": shutil.which("llama-server") is not None,
        "ollama": shutil.which("ollama") is not None,
    }
    return {
        "platform": system,
        "architecture": machine,
        "engine_policy": select_engines(system=system, machine=machine),
        "detected_engines": binaries,
    }
