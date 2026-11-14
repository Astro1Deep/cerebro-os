#!/usr/bin/env python3
"""Diagnóstico no destructivo de Cerebro OS 1.5."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from cerebro_core.local_endpoint import configured_local_ai_endpoint, local_http_open, models_endpoint
from cerebro_runtime import load_runtime_config
from cerebro_core.beatos import beatos_api_base, library_status

REQUIRED_FILES = ("composer_server.py", "composer_ui.html", "VERSION")
REQUIRED_MODULES = ("aiohttp", "fastapi", "pydantic", "uvicorn")
JARVIS_HEALTH_ENDPOINT = "http://127.0.0.1:8083/api/health"
MLX_ERROR_LOG = Path("logs/mlx_server_error.log")
MLX_BOOT_MARKER = "CEREBRO_MLX_RUN_START"
MLX_GENERATION_FAILURE_SIGNATURES = (
    "Exception in thread Thread-1 (_generate)",
    "ValueError: [broadcast_shapes]",
    "Command buffer execution failed: Insufficient Memory",
)
MLX_GENERATION_SUCCESS_SIGNATURE = '"POST /v1/chat/completions HTTP/1.1" 200'
SERVICE_AGENTS = {
    "ai": "com.ivan.mlx.server",
    "composer": "com.ivan.mlx.composer",
    "jarvis": "com.ivan.cerebro.jarvis",
    "telegram": "com.ivan.cerebro.telegram",
    "mayordomo": "com.ivan.cerebro.mayordomo",
}
SERVICE_AGENT_FILES = {
    "ai": "com.ivan.mlx.server.plist",
    "composer": "com.ivan.mlx.composer.plist",
    "jarvis": "com.ivan.mlx.jarvis.plist",
    "telegram": "com.ivan.mlx.telegram.plist",
    "mayordomo": "com.ivan.cerebro.mayordomo.plist",
}


def probe_json(url: str, timeout: float = 3.0) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": "CerebroDoctor/1.5"})
    for _attempt in range(2):
        try:
            with local_http_open(request, timeout=timeout) as response:
                if response.status != 200:
                    return None
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
            continue
    return None


def mlx_generation_failure_from_log(error_log: Path) -> bool:
    """Detect a generation crash from the current MLX run only.

    Launchd appends to its standard-error file across restarts.  The boot marker
    therefore creates a safe boundary: an old crash cannot make a healthy new
    process restart again.
    """

    try:
        with error_log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    marker_index = text.rfind(MLX_BOOT_MARKER)
    if marker_index < 0:
        return False
    current_run = text[marker_index + len(MLX_BOOT_MARKER) :]
    return any(signature in current_run for signature in MLX_GENERATION_FAILURE_SIGNATURES)


def mlx_inference_state_from_log(error_log: Path) -> str:
    """Return the known generation state for the current MLX process only."""

    try:
        with error_log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return "unverified"
    marker_index = text.rfind(MLX_BOOT_MARKER)
    if marker_index < 0:
        return "unverified"
    current_run = text[marker_index + len(MLX_BOOT_MARKER) :]
    if any(signature in current_run for signature in MLX_GENERATION_FAILURE_SIGNATURES):
        return "failed"
    return "ready" if MLX_GENERATION_SUCCESS_SIGNATURE in current_run else "unverified"


def launch_agent_state(label: str) -> str:
    """Return a conservative local launchd state without changing a service."""

    if platform.system() != "Darwin":
        return "not_supported"
    target = f"gui/{os.getuid()}/{label}"
    try:
        completed = subprocess.run(
            ["launchctl", "print", target],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "not_available"
    if completed.returncode != 0:
        return "not_loaded"
    return "running" if "state = running" in completed.stdout else "not_running"


def mlx_auxiliary_listener_state() -> str:
    """Check whether the MLX process exposes unnecessary MPI listeners."""

    if platform.system() != "Darwin":
        return "not_supported"
    try:
        listeners = subprocess.run(
            ["lsof", "-nP", "-iTCP:1024", "-iTCP:1025", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "not_checked"
    pids = {line[1:] for line in listeners.stdout.splitlines() if line.startswith("p") and line[1:].isdigit()}
    for pid in pids:
        try:
            command = subprocess.run(
                ["ps", "-p", pid, "-o", "command="],
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "mlx_lm.server" in command:
            return "exposed"
    return "closed"


def mayordomo_state(base_dir: Path) -> str:
    """Report the proposal-only watcher state without starting or stopping it."""

    del base_dir
    return (
        "review_watcher_active"
        if launch_agent_state(SERVICE_AGENTS["mayordomo"]) == "running"
        else "review_watcher_inactive"
    )


def restart_launch_agent(label: str, agent_file: Path) -> dict[str, object]:
    """Restart one known local LaunchAgent only after an explicit repair request."""

    if platform.system() != "Darwin":
        return {"attempted": False, "reason": "launchd_no_disponible"}
    target = f"gui/{os.getuid()}/{label}"
    try:
        completed = subprocess.run(
            ["launchctl", "kickstart", "-k", target],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"attempted": True, "restarted": False, "reason": type(error).__name__}
    if completed.returncode == 0:
        return {"attempted": True, "restarted": True, "reason": "kickstart_ok"}

    if not agent_file.is_file():
        return {"attempted": True, "restarted": False, "reason": "agent_file_missing"}
    try:
        fallback = subprocess.run(
            ["launchctl", "load", "-w", str(agent_file)],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"attempted": True, "restarted": False, "reason": type(error).__name__}
    return {
        "attempted": True,
        "restarted": fallback.returncode == 0,
        "reason": "load_ok" if fallback.returncode == 0 else "launchctl_failed",
    }


def repair_local_services(report: dict) -> dict[str, dict[str, object]]:
    """Repair confirmed-offline supervised services; never move user files."""

    states = {
        "ai": (
            bool(report.get("ai_online"))
            and report.get("ai_inference_state", "unverified") != "failed"
            and report.get("ai_auxiliary_listener_state", "closed") != "exposed"
        ),
        "composer": bool(report.get("composer_online")),
        "jarvis": bool(report.get("jarvis_online")),
        "telegram": report.get("telegram_state") == "running",
        "mayordomo": report.get("mayordomo_state") == "review_watcher_active",
    }
    results: dict[str, dict[str, object]] = {}
    for service, online in states.items():
        if online:
            results[service] = {"attempted": False, "reason": "already_online"}
            continue
        results[service] = restart_launch_agent(
            SERVICE_AGENTS[service],
            Path.home() / "Library" / "LaunchAgents" / SERVICE_AGENT_FILES[service],
        )

    return results


def build_report(
    base_dir: Path,
    *,
    probe_network: bool = True,
    required_modules: Iterable[str] = REQUIRED_MODULES,
) -> dict:
    base_dir = base_dir.resolve()
    missing_files = [name for name in REQUIRED_FILES if not (base_dir / name).is_file()]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]

    composer_health = None
    ai_health = None
    ai_inference_state = "not_checked"
    jarvis_health = None
    telegram_state = "not_checked"
    ai_auxiliary_listener_state = "not_checked"
    gateway_online = False
    circuit_data = None
    mpc_server_online = False
    disks_status = {}
    if probe_network:
        try:
            configured_endpoint = load_runtime_config(base_dir).get("CEREBRO_AI_URL")
        except (OSError, ValueError):
            configured_endpoint = None
        composer_health = probe_json("http://127.0.0.1:8082/api/health")
        ai_health = probe_json(
            models_endpoint(
                configured_local_ai_endpoint(os.environ.get("CEREBRO_AI_URL") or configured_endpoint)
            )
        )
        ai_inference_state = (
            "offline"
            if ai_health is None
            else mlx_inference_state_from_log(base_dir / MLX_ERROR_LOG)
        )
        jarvis_health = probe_json(JARVIS_HEALTH_ENDPOINT)
        telegram_state = launch_agent_state(SERVICE_AGENTS["telegram"])
        ai_auxiliary_listener_state = mlx_auxiliary_listener_state()

        # Inmunología Microservicios: Gateway :8081 & Watchdog Circuit Breaker
        circuit_data = probe_json("http://127.0.0.1:8081/v1/circuits", timeout=2.0)
        gateway_online = circuit_data is not None

        mpc_server_online = probe_json(f"{beatos_api_base()}/api/status", timeout=1.5) is not None
        disks_status = {item["key"]: item["mounted"] for item in library_status()["items"]}

    errors = []
    warnings = []
    if sys.version_info < (3, 11):
        errors.append("Python 3.11 o superior es obligatorio")
    if missing_files:
        errors.append("Faltan archivos principales")
    if missing_modules:
        errors.append("Faltan dependencias Python")
    if probe_network and composer_health is None:
        warnings.append("El servicio del compositor no responde en 127.0.0.1:8082")
    if probe_network and ai_health is None:
        warnings.append("El endpoint configurado de IA local no responde; el análisis musical básico seguirá disponible")
    if probe_network and ai_inference_state == "failed":
        warnings.append("MLX respondió al transporte, pero su hebra de generación falló; usa doctor --repair")
    if probe_network and ai_auxiliary_listener_state == "exposed":
        warnings.append("MLX expone listeners distribuidos innecesarios; usa doctor --repair tras aplicar la configuración local")
    if probe_network and jarvis_health is None:
        warnings.append("Jarvis no responde en 127.0.0.1:8083")

    status = "error" if errors else "degraded" if warnings else "ok"

    watcher_state = mayordomo_state(base_dir) if probe_network else "not_checked"
    return {
        "status": status,
        "version": (base_dir / "VERSION").read_text(encoding="utf-8").strip()
        if (base_dir / "VERSION").is_file()
        else "desconocida",
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "base_dir": str(base_dir),
        "missing_files": missing_files,
        "missing_modules": missing_modules,
        "composer_online": composer_health is not None,
        "ai_online": ai_health is not None,
        "ai_inference_state": ai_inference_state,
        "ai_auxiliary_listener_state": ai_auxiliary_listener_state,
        "jarvis_online": jarvis_health is not None,
        "telegram_state": telegram_state,
        "mayordomo_state": watcher_state,
        "gateway_online": gateway_online,
        "circuit_breaker": circuit_data.get("circuits") if circuit_data else None,
        "gateway_failover_active": circuit_data.get("failover_active", False) if circuit_data else False,
        "mpc_server_online": mpc_server_online,
        "disks_status": disks_status,
        "errors": errors,
        "warnings": warnings,
    }


def exit_code_for_report(report: dict) -> int:
    return 0 if report.get("status") == "ok" else 1


def print_human(report: dict) -> None:
    ok = report["status"] == "ok"
    print("CEREBRO OS - DIAGNÓSTICO v1.5")
    print("=" * 42)
    print(f"Estado principal : {'OK' if ok else report['status'].upper()}")
    print(f"Versión          : {report['version']}")
    print(f"Sistema          : {report['platform']} ({report['architecture']})")
    print(f"Python           : {report['python']}")
    print(f"Instalación      : {report['base_dir']}")
    print(f"Compositor 8082  : {'ONLINE' if report['composer_online'] else 'OFFLINE/no comprobado'}")
    print(f"IA local 8080    : {'ONLINE' if report['ai_online'] else 'OFFLINE/no comprobado'}")
    inference_state = report.get("ai_inference_state", "not_checked")
    if inference_state == "failed":
        print("Generación MLX  : FALLÓ (doctor --repair puede reiniciarla)")
    elif inference_state == "ready":
        print("Generación MLX  : LISTA (respuesta verificada en el arranque actual)")
    elif inference_state == "unverified":
        print("Generación MLX  : SIN PRUEBA ACTIVA (no se fuerza una carga durante doctor)")
    listener_state = report.get("ai_auxiliary_listener_state", "not_checked")
    if listener_state == "exposed":
        print("Red auxiliar MLX: EXPUESTA (doctor --repair puede reiniciar con backend local)")
    elif listener_state == "closed":
        print("Red auxiliar MLX: CERRADA")
    print(f"Jarvis 8083      : {'ONLINE' if report['jarvis_online'] else 'OFFLINE/no comprobado'}")
    if "gateway_online" in report and report["gateway_online"]:
        cb = report.get("circuit_breaker") or {}
        mlx_st = cb.get("mlx", {}).get("state", "CLOSED")
        lat_ms = cb.get("mlx", {}).get("last_latency_ms", 0.0)
        failover = " (FAILOVER ACTIVO A OLLAMA)" if report.get("gateway_failover_active") else ""
        print(f"Gateway 8081     : ONLINE [Circuito: {mlx_st} | Latencia: {lat_ms:.1f}ms{failover}]")
    if "mpc_server_online" in report:
        print(f"BeatOS/MPC       : {'ONLINE' if report['mpc_server_online'] else 'STANDBY/OFFLINE'}")
    if report.get("disks_status"):
        mounted_count = sum(1 for v in report["disks_status"].values() if v)
        total_count = len(report["disks_status"])
        print(f"Discos Producción: {mounted_count}/{total_count} montados")
    print(f"Telegram         : {report['telegram_state'].upper()}")
    mayordomo_label = (
        "ACTIVO: PROPUESTAS CON APROBACIÓN"
        if report["mayordomo_state"] == "review_watcher_active"
        else "INACTIVO: REVISIÓN SEGURA"
    )
    print(f"Mayordomo        : {mayordomo_label}")
    if report["missing_files"]:
        print("Archivos ausentes:", ", ".join(report["missing_files"]))
    if report["missing_modules"]:
        print("Módulos ausentes :", ", ".join(report["missing_modules"]))
    for warning in report["warnings"]:
        print("AVISO:", warning)
    for error in report["errors"]:
        print("ERROR:", error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de Cerebro OS")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--offline", action="store_true", help="No consultar puertos locales")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Reiniciar sólo servicios locales supervisados que estén caídos",
    )
    parser.add_argument("--json", action="store_true", help="Emitir JSON")
    args = parser.parse_args(argv)
    if args.offline and args.repair:
        parser.error("--offline y --repair no se pueden combinar")

    report = build_report(args.base_dir, probe_network=not args.offline)
    if args.repair:
        try:
            repair = repair_local_services(report)
            report = build_report(args.base_dir, probe_network=not args.offline)
            report["repair"] = repair
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print_human(report)
            return 0
        except Exception as e:
            if args.json:
                print(json.dumps({"status": "error", "error": str(e)}))
            else:
                print(f"Error al reparar: {e}", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return exit_code_for_report(report)



if __name__ == "__main__":
    raise SystemExit(main())
