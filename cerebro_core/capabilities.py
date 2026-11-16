"""Política portable de selección de motores locales."""

from __future__ import annotations

import platform
import shutil
import json
import urllib.request

from cerebro_core.local_endpoint import local_http_open

APPLE_SILICON_ARCHITECTURES = {"arm64", "aarch64"}


def probe_ollama_loaded_models() -> dict[str, object]:
    """Read local Ollama's loaded-model report without launching or loading models."""
    request = urllib.request.Request("http://127.0.0.1:11434/api/ps", method="GET")
    try:
        with local_http_open(request, timeout=2.0) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("response too large")
        payload = json.loads(data)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("invalid response")
        models = []
        for model in payload["models"][:100]:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                continue
            def optional_nonnegative_int(value):
                return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
            details = model.get("details") if isinstance(model.get("details"), dict) else {}
            models.append({
                "name": model["name"][:256],
                "digest": model.get("digest") if isinstance(model.get("digest"), str) else None,
                "loaded_bytes": optional_nonnegative_int(model.get("size")),
                "vram_bytes": optional_nonnegative_int(model.get("size_vram")),
                "context_length": optional_nonnegative_int(model.get("context_length")),
                "quantization": details.get("quantization_level") if isinstance(details.get("quantization_level"), str) else None,
            })
        return {"api_reachable": True, "loaded_models": models,
                "generation_verified": False, "gpu_execution_verified": False}
    except (OSError, ValueError, TypeError):
        return {"api_reachable": False, "loaded_models": [],
                "generation_verified": False, "gpu_execution_verified": False}


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
