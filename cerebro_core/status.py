"""Estado de sólo lectura para el núcleo local de Cerebro."""

from __future__ import annotations

import json
import os
import platform
import urllib.error
import urllib.request
from pathlib import Path

from cerebro_core.hermes import inspect_hermes
from cerebro_core.local_endpoint import configured_local_ai_endpoint, local_http_open, models_endpoint
from cerebro_runtime import load_runtime_config


COMPOSER_ENDPOINT = "http://127.0.0.1:8082/api/health"
JARVIS_ENDPOINT = "http://127.0.0.1:8083/api/health"


def _hermes_integration_state(base_dir: Path) -> dict[str, object]:
    resolved_base = base_dir.resolve()
    default_stack = (Path.home() / ".mlx-stack").resolve()
    hermes_home = (Path.home() / ".hermes") if resolved_base == default_stack else (resolved_base / ".hermes")
    info = inspect_hermes(hermes_home)
    connector = info.get("connector", {})
    hermes_meta = info.get("hermes", {})
    if isinstance(connector, dict) and connector.get("state") == "configured":
        return {"state": "configured", "target": connector.get("target", "http://127.0.0.1:8081/v1")}
    if isinstance(hermes_meta, dict) and hermes_meta.get("config_present"):
        return {"state": "detected"}
    return {"state": "not_configured"}


def service_endpoints(base_dir: Path | None = None) -> dict[str, str]:
    """Return only validated loopback health endpoints for the current runtime."""

    configured_endpoint: str | None = None
    if base_dir is not None:
        try:
            configured_endpoint = load_runtime_config(base_dir).get("CEREBRO_AI_URL")
        except (OSError, ValueError):
            configured_endpoint = None
    ai_endpoint = configured_local_ai_endpoint(os.environ.get("CEREBRO_AI_URL") or configured_endpoint)
    return {
        "ai": models_endpoint(ai_endpoint),
        "composer": COMPOSER_ENDPOINT,
        "jarvis": JARVIS_ENDPOINT,
    }


def _service_state(endpoint: str, *, probe_network: bool) -> dict[str, str]:
    if not probe_network:
        return {"state": "not_checked", "endpoint": endpoint}

    request = urllib.request.Request(endpoint, headers={"User-Agent": "CerebroCLI/0.1"})
    state = "offline"
    for _attempt in range(2):
        try:
            with local_http_open(request, timeout=3.0) as response:
                response.read(1)
                state = "online" if response.status == 200 else "degraded"
                break
        except (OSError, urllib.error.URLError):
            continue
    return {"state": state, "endpoint": endpoint}


def build_status(base_dir: Path, *, probe_network: bool) -> dict[str, object]:
    """Build a side-effect-free status report for a Cerebro installation."""
    base_dir = base_dir.resolve()
    version_file = base_dir / "VERSION"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else "unknown"
    services = {
        name: _service_state(endpoint, probe_network=probe_network)
        for name, endpoint in service_endpoints(base_dir).items()
    }

    # Telemetría de Failover & Circuit Breakers (Gateway :8081)
    circuits_data = {}
    if probe_network:
        try:
            req = urllib.request.Request("http://127.0.0.1:8081/v1/circuits", headers={"User-Agent": "CerebroCLI/0.1"})
            with local_http_open(req, timeout=1.0) as resp:
                circuits_data = json.loads(resp.read().decode())
        except Exception:
            circuits_data = {"status": "unavailable"}

    # Estado de FAISS L1 Hot-Cache
    faiss_state = "offline"
    try:
        from cerebro_core.vector_engine import _FAISS_AVAILABLE
        faiss_state = "online (l1-simd)" if _FAISS_AVAILABLE else "fallback-lancedb"
    except Exception:
        faiss_state = "not_loaded"

    # Estado de AutoDoctor Sentinel
    autodoctor_state = "offline"
    try:
        import subprocess
        if not probe_network or platform.system() != "Darwin":
            return_status = None
        else:
            return_status = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=1.0)
        if return_status is not None and "com.cerebro.autodoctor" in return_status.stdout:
            autodoctor_state = "active (launchagent)"
    except Exception:
        pass

    return {
        "schema_version": 1,
        "version": version,
        "local_first": True,
        "runtime": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "services": services,
        "circuits": circuits_data,
        "faiss_cache": faiss_state,
        "autodoctor": autodoctor_state,
        "integrations": {"hermes": _hermes_integration_state(base_dir)},
    }


def render_status(report: dict[str, object], *, as_json: bool) -> str:
    if as_json:
        return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)

    services = report["services"]
    assert isinstance(services, dict)
    
    lines = [
        f"Cerebro {report['version']} (local-first / Apple M4 Pro)",
        f"IA local:     {services['ai']['state']}",
        f"Composer:     {services['composer']['state']}",
        f"Jarvis:       {services['jarvis']['state']}",
        f"Hermes:       {report['integrations']['hermes']['state']}",
        f"FAISS L1:     {report.get('faiss_cache', 'unknown')}",
        f"AutoDoctor:   {report.get('autodoctor', 'unknown')}",
    ]

    circuits = report.get("circuits", {})
    if isinstance(circuits, dict) and circuits.get("status") == "ok":
        mlx_c = circuits.get("circuits", {}).get("mlx", {})
        ol_c = circuits.get("circuits", {}).get("ollama", {})
        fo_active = circuits.get("failover_active", False)
        fo_str = "ACTIVADO (conmutado a contingencia)" if fo_active else "STANDBY (ruta normal MLX)"
        lines.append(f"Failover:     {fo_str}")
        lines.append(f"  ├─ MLX :8080 Circuit:     {mlx_c.get('state', 'unknown')} (lat: {mlx_c.get('last_latency_ms', 0)}ms)")
        lines.append(f"  └─ Ollama :11434 Circuit: {ol_c.get('state', 'unknown')} (lat: {ol_c.get('last_latency_ms', 0)}ms)")

    return "\n".join(lines)
