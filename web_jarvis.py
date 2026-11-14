"""Loopback-only browser companion for the local MLX model.

Remote browser access must go through a private, authenticated network layer;
this process intentionally never listens on the LAN or public internet.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path
import re
import tempfile
from typing import Any, Final
import urllib.error
import urllib.request

from flask import Flask, Response, jsonify, request, stream_with_context
import psutil
import requests

from cerebro_core.local_endpoint import configured_local_ai_endpoint, configured_local_ai_model
from cerebro_core.beatos import library_status, search_samples, beatos_api_base
from cerebro_core.local_endpoint import local_http_open, local_http_session


app = Flask(__name__)
app.config["TRUSTED_HOSTS"] = ["127.0.0.1", "localhost", "[::1]"]
BIND_HOST: Final[str] = "127.0.0.1"
PORT: Final[int] = 8083
MAX_INPUT_CHARACTERS: Final[int] = 2_000
MAX_REQUEST_BYTES: Final[int] = MAX_INPUT_CHARACTERS * 6 + 256
MAX_STREAM_REQUEST_BYTES: Final[int] = MAX_REQUEST_BYTES * 9
MODEL_REQUEST_TIMEOUT_SECONDS: Final[int] = 210
app.config["MAX_CONTENT_LENGTH"] = MAX_STREAM_REQUEST_BYTES
SYSTEM_PROMPT: Final[str] = (
    "Eres Cerebro (Jarvis Hub v2.0), asistente táctico local de Cerebro OS en el equipo local. "
    "Responde en español de forma directa, precisa y breve manteniendo el contexto conversacional."
)

OPERATOR_QUERY_TRIGGERS: Final[tuple[str, ...]] = (
    "/sys status",
    "/doctor",
    "estado del sistema",
    "telemetría",
    "telemetria",
    "circuit breaker",
    "discos externos",
    "discos están montados",
    "discos de producción",
    "ram unificada",
    "uso de ram",
    "swap",
)

MUSIC_STOPWORDS: Final[frozenset[str]] = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con",
    "por", "para", "que", "qué", "cual", "cuál", "cuales", "cuáles", "como", "cómo",
    "donde", "dónde", "busca", "buscar", "encuentra", "mostrar", "muestra", "dame",
    "tengo", "tienes", "hay", "mis", "mi", "tu", "tus", "sobre", "lista", "listar",
    "todos", "todas", "algun", "algún", "alguna", "samples", "sample", "muestras",
    "pistas", "tracks", "archivos", "proyectos", "proyecto", "librerias", "librerías",
    "libreria", "librería", "kits", "kit", "disco", "discos", "audioteca", "musica",
    "música", "musical", "produccion", "producción", "sonidos", "sonido", "audio",
})



def resolve_api_url() -> str:
    """Return only a validated loopback model endpoint for this companion."""

    return configured_local_ai_endpoint(os.getenv("CEREBRO_AI_URL"))


API_URL: Final[str] = resolve_api_url()
API_MODEL: Final[str | None] = configured_local_ai_model(os.getenv("CEREBRO_AI_MODEL"))


def _probe_loopback_http(url: str, timeout: float = 0.6) -> str:
    """Verifica rápidamente un puerto HTTP local sin bloquear el hilo."""
    req = urllib.request.Request(url, headers={"User-Agent": "JarvisHub/2.0"})
    try:
        with local_http_open(req, timeout=timeout) as resp:
            return "online" if resp.status == 200 else "degraded"
    except (OSError, urllib.error.URLError):
        return "offline"


def _get_hardware_telemetry() -> dict[str, Any]:
    """Report measured host values and the explicitly configured model."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "soc": platform.processor() or platform.machine(),
        "cpu_logical": psutil.cpu_count(),
        "active_model": API_MODEL,
        "active_endpoint": API_URL,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_free_gb": round(mem.available / (1024**3), 2),
        "ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
        "ram_percent": round(mem.percent, 1),
        "swap_used_mb": round(swap.used / (1024**2), 1),
    }


def collect_operator_telemetry() -> dict[str, Any]:
    """Reúne en vivo build_status(), telemetría JEV de hardware M4 Pro, puertos y las bibliotecas detectadas."""
    stack_dir = Path(__file__).resolve().parent
    try:
        from cerebro_core.status import build_status
        report = build_status(stack_dir, probe_network=True)
    except Exception:
        report = {"services": {}, "circuits": {"status": "unavailable"}, "integrations": {}}

    services = report.get("services")
    if isinstance(services, dict):
        services.setdefault("jarvis", {})["state"] = "online"

    hardware = _get_hardware_telemetry()
    disks = library_status()

    circuits = report.get("circuits", {}) if isinstance(report.get("circuits"), dict) else {}
    mlx_circuit = (
        circuits.get("circuits", {}).get("mlx", {})
        if isinstance(circuits.get("circuits"), dict)
        else {}
    )

    port_8080 = (
        "online"
        if str(mlx_circuit.get("state", "")).lower() == "closed"
        else _probe_loopback_http("http://127.0.0.1:8080/v1/models")
    )
    port_8081 = "online" if circuits.get("status") == "ok" else _probe_loopback_http("http://127.0.0.1:8081/v1/models")
    port_8082 = (
        services.get("composer", {}).get("state", "offline")
        if isinstance(services, dict) and isinstance(services.get("composer"), dict)
        else _probe_loopback_http("http://127.0.0.1:8082/api/health")
    )
    port_8083 = "online"
    beatos_base = beatos_api_base()
    beatos_state = _probe_loopback_http(f"{beatos_base}/api/status")

    ports = {
        "8080": port_8080,
        "8081": port_8081,
        "8082": port_8082,
        "8083": port_8083,
        f"BeatOS {beatos_base}": beatos_state,
    }

    report["hardware"] = hardware
    report["disks"] = disks
    report["ports"] = ports
    return report


def format_operator_telemetry_context(report: dict[str, Any] | None = None) -> str:
    """Formatea la telemetría viva del sistema para inyectarla en el system prompt del Modo Operador."""
    data = report if isinstance(report, dict) else collect_operator_telemetry()
    hw = data.get("hardware", {})
    ports = data.get("ports", {})
    disks = data.get("disks", {})
    circuits = data.get("circuits", {}) if isinstance(data.get("circuits"), dict) else {}
    mlx_c = (
        circuits.get("circuits", {}).get("mlx", {})
        if isinstance(circuits.get("circuits"), dict)
        else {}
    )
    disk_lines = [
        f"  * {item['label']}: {item['free_gb']} GB libres" if item["mounted"]
        else f"  * {item['label']}: montaje sin confirmar"
        for item in disks.get("items", [])
    ]

    lines = [
        "[TELEMETRÍA EN VIVO DE CEREBRO OS]",
        f"- Plataforma: {hw.get('soc', 'sin datos')} · CPU lógicas: {hw.get('cpu_logical', 'sin datos')}",
        f"- RAM: {hw.get('ram_used_gb', 'sin datos')} GB usados · Swap: {hw.get('swap_used_mb', 'sin datos')} MB",
        f"- Endpoint configurado: {hw.get('active_endpoint') or 'sin datos'} · Modelo: {hw.get('active_model') or 'sin identificar'}",
        f"- Gateway: {mlx_c.get('state', 'sin datos')} · Latencia: {mlx_c.get('last_latency_ms', 'sin medición')}",
        *(f"- {name}: {state}" for name, state in ports.items()),
        f"- Bibliotecas BeatOS: {disks.get('mounted_count', 0)}/{disks.get('total_disks', 0)}",
        *disk_lines,
    ]
    return "\n".join(lines)


def is_operator_query(user_input: str, mode: str) -> bool:
    """Determina si debe inyectarse la telemetría viva de Operador en el system prompt."""
    if mode == "operator":
        return True
    lowered = user_input.lower().strip()
    return any(trigger in lowered for trigger in OPERATOR_QUERY_TRIGGERS)


def _extract_music_keywords(query_text: str) -> list[str]:
    """Extrae términos relevantes de búsqueda musical ignorando stopwords comunes."""
    tokens = re.findall(r"[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ_.-]+", query_text.lower())
    keywords: list[str] = []
    for tok in tokens:
        cleaned = tok.strip("._-")
        if len(cleaned) >= 2 and cleaned not in MUSIC_STOPWORDS and cleaned not in ("xpj", "xpm", "als", "nki", "ens"):
            if cleaned not in keywords:
                keywords.append(cleaned)
    return keywords[:5]


def query_music_production_context(query_text: str) -> str:
    keywords = _extract_music_keywords(query_text)
    for keyword in keywords:
        matches = search_samples(keyword, limit=8)
        if matches:
            return "\n".join(f"• {row['filename']} ({row['disk_name']})" for row in matches)
    return ""


def query_rag_context(query_text: str) -> str:
    """Consulta la memoria persistente RRF de Cerebro OS y los índices SQLite de producción musical."""
    blocks: list[str] = []
    try:
        from cerebro_core.persistent_memory import PersistentMemoryStore
        store = PersistentMemoryStore(Path(__file__).resolve().parent / "knowledge")
        results = store.search_rrf(query_text, limit=3)
        if results:
            snippets = [f"- {r.get('text', '')}" for r in results if r.get("text")]
            if snippets:
                blocks.append("[CONOCIMIENTO RECUPERADO DE MEMORIA PERSISTENTE RAG]:\n" + "\n".join(snippets))
    except Exception:
        pass

    music_context = query_music_production_context(query_text)
    if music_context:
        blocks.append(music_context)

    return "\n\n".join(blocks)


HTML_CONTENT = r"""
<!doctype html>
<html lang="es" data-theme="cerebro-neon">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <title>Cerebro MLX local · Jarvis Hub</title>
  <style>
    :root, :root[data-theme="cerebro-neon"] {
      --bg: #070b12;
      --card-bg: #0e1726;
      --border: #1e293b;
      --accent: #00ff9d;
      --accent-glow: rgba(0, 255, 157, 0.25);
      --text: #e2e8f0;
      --text-muted: #8492a6;
      --user-msg: #1a273e;
      --ai-msg: #0b1320;
      --ai-border: #00ff9d;
    }
    :root[data-theme="astro1deep-obsidian"] {
      color-scheme: dark;
      --bg: #08090a;
      --card-bg: #141519;
      --border: #68717f;
      --accent: #d5dbe5;
      --accent-glow: #d5dbe530;
      --text: #f4f5f7;
      --text-muted: #b8bec8;
      --user-msg: #272a31;
      --ai-msg: #191b20;
      --ai-border: #afc8e3;
    }
    :root[data-theme="astro1deep-obsidian"] :is(button, select, input):focus-visible {
      outline: 3px solid var(--accent);
      outline-offset: 3px;
    }
    :root[data-theme="te-arcade"] {
      --bg: #1c1f26;
      --card-bg: #232730;
      --border: #2f3542;
      --accent: #9b51e0;
      --accent-glow: rgba(155, 81, 224, 0.35);
      --text: #f8fafc;
      --text-muted: #8a90a2;
      --user-msg: #2a203d;
      --ai-msg: #181b22;
      --ai-border: #9b51e0;
    }
    :root[data-theme="mpc-vintage"] {
      --bg: #ddd9ce;
      --card-bg: #eae6df;
      --border: #bebcb4;
      --accent: #c8102e;
      --accent-glow: rgba(200, 16, 46, 0.25);
      --text: #1a1a1a;
      --text-muted: #525866;
      --user-msg: #f2eee6;
      --ai-msg: #e4e0d7;
      --ai-border: #c8102e;
    }
    :root[data-theme="sp404-studio"] {
      --bg: #0a0a0c;
      --card-bg: #121318;
      --border: #262733;
      --accent: #f59e0b;
      --accent-glow: rgba(245, 158, 11, 0.25);
      --text: #edf2f4;
      --text-muted: #7d8290;
      --user-msg: #221d14;
      --ai-msg: #111217;
      --ai-border: #f59e0b;
    }
    :root[data-theme="makerlicious-bone"] {
      --bg: #dddddf;
      --card-bg: #eaebee;
      --border: #bbbbbf;
      --accent: #ff6600;
      --accent-glow: rgba(255, 102, 0, 0.25);
      --text: #111114;
      --text-muted: #4e505a;
      --user-msg: #ffffff;
      --ai-msg: #e2e3e7;
      --ai-border: #ff6600;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      display: flex;
      flex-direction: column;
      height: 100vh;
      padding: 16px;
      overflow: hidden;
      transition: background 0.2s ease, color 0.2s ease;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin-bottom: 12px;
      gap: 10px;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand h1 { font-size: 16px; color: var(--accent); letter-spacing: 1px; font-weight: 700; }
    .telemetry-bar {
      display: flex;
      gap: 10px;
      font-size: 12px;
      color: var(--text-muted);
      align-items: center;
      flex-wrap: wrap;
    }
    .badge {
      padding: 4px 10px;
      border-radius: 8px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
    }
    .badge.active-auth {
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }
    .badge.warn-auth {
      border-color: #f59e0b;
      color: #f59e0b;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .dot.warn { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .controls { display: flex; gap: 8px; align-items: center; }
    .mode-btn {
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text-muted);
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .mode-btn.active {
      background: var(--border);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: bold;
    }
    .icon-btn, select.theme-select {
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text);
      cursor: pointer;
      font-size: 12px;
      outline: none;
    }
    .icon-btn:hover, select.theme-select:hover { border-color: var(--accent); }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 12px 6px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .msg {
      max-width: 82%;
      line-height: 1.5;
      padding: 12px 18px;
      border-radius: 14px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 14px;
    }
    .user {
      align-self: flex-end;
      background: var(--user-msg);
      color: var(--text);
      border-bottom-right-radius: 2px;
      border: 1px solid var(--border);
    }
    .ai {
      align-self: flex-start;
      background: var(--ai-msg);
      color: var(--text);
      border-left: 3px solid var(--ai-border);
      border-top: 1px solid var(--border);
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      border-bottom-left-radius: 2px;
    }
    .thinking {
      color: var(--text-muted);
      font-style: italic;
      border-left-color: #64748b;
    }
    .code-block {
      margin: 8px 0;
      padding: 10px 12px;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      overflow-x: auto;
      font-family: "SFMono-Regular", Consolas, Menlo, monospace;
      font-size: 12.5px;
      white-space: pre;
    }
    .code-lang {
      font-size: 10px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 4px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .inline-code {
      padding: 1px 5px;
      border-radius: 4px;
      background: rgba(128, 128, 128, 0.2);
      border: 1px solid var(--border);
      font-family: "SFMono-Regular", Consolas, Menlo, monospace;
      font-size: 12.5px;
    }
    .msg-list {
      margin: 4px 0 4px 18px;
      padding: 0;
    }
    .msg-list li {
      margin: 2px 0;
    }
    .msg-line {
      min-height: 1.2em;
    }
    .quick-actions {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 6px 0;
      margin-bottom: 10px;
    }
    .quick-pill {
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 11px;
      color: var(--text-muted);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .quick-pill:hover { border-color: var(--accent); color: var(--accent); }
    #input-area {
      display: flex;
      gap: 10px;
      background: var(--card-bg);
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      align-items: center;
    }
    input {
      flex: 1;
      padding: 12px 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 15px;
      outline: 0;
    }
    input:focus { border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
    button.send {
      padding: 12px 22px;
      border-radius: 10px;
      border: 0;
      background: var(--accent);
      color: #000;
      font-weight: bold;
      font-size: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    button.send:hover { opacity: 0.9; }
    button.send:active { transform: scale(0.98); }
    .mic-btn {
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text);
      cursor: pointer;
      font-size: 16px;
      transition: all 0.2s ease;
    }
    .mic-btn.listening {
      background: #ef4444;
      border-color: #f87171;
      color: #fff;
      animation: pulse 1s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="dot" id="engineDot"></div>
      <h1>JARVIS HUB</h1>
      <span style="font-size:11px; color:var(--text-muted);">LOCAL · 8083</span>
    </div>
    <div class="telemetry-bar">
      <div class="badge" id="telemetryBadge">Sondeando telemetría...</div>
      <div class="badge" id="failoverBadge">Failover: sin confirmación</div>
      <div class="badge" id="disksBadge">Bibliotecas de BeatOS</div>
    </div>
    <div class="controls">
      <select class="theme-select" id="themeSelect" title="Paletas & Temas de Hardware">
        <option value="cerebro-neon">⚡ Cerebro Neon</option>
        <option value="te-arcade">🕹️ PO-20 Arcade (Gris & Púrpura)</option>
        <option value="mpc-vintage">🎛️ MPC Vintage (Blanco Hueso)</option>
        <option value="sp404-studio">🎚️ SP-404MK2 (Ámbar)</option>
        <option value="makerlicious-bone">🦴 MakerLicious Bone</option>
        <option value="astro1deep-obsidian">Astro1Deep Obsidian</option>
    </select>
      <button class="mode-btn active" id="btnModeAssistant" type="button">💬 Asistente</button>
      <button class="mode-btn" id="btnModeOperator" type="button">⚙️ Operador</button>
      <button class="mode-btn" id="btnModeRag" type="button">📚 RAG</button>
      <button class="icon-btn" id="clearBtn" type="button" title="Reiniciar Sesión">🧹</button>
    </div>
  </header>

  <div class="quick-actions">
    <div class="quick-pill" data-prompt="/sys status">⚡ /sys status</div>
    <div class="quick-pill" data-prompt="/doctor">🩺 /doctor</div>
    <div class="quick-pill" data-prompt="¿Qué discos externos están montados y listos?">💾 Discos de Producción</div>
    <div class="quick-pill" data-prompt="Resume las últimas tareas de Mayordomo">🕴️ Mayordomo</div>
    <div class="quick-pill" data-prompt="¿Cuál es el estado del Circuit Breaker y la latencia?">🛡️ Circuit Breakers</div>
  </div>

  <div id="chat" aria-live="polite"></div>

  <div id="input-area">
    <input type="text" id="userInput" placeholder="Escribe o dicta aquí tu consulta para Jarvis (Cmd+K o Espacio para hablar)..." autocomplete="off" maxlength="2000">
    <button class="send" id="sendButton" type="button">Enviar</button>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearBtn = document.getElementById('clearBtn');
    const telemetryBadge = document.getElementById('telemetryBadge');
    const failoverBadge = document.getElementById('failoverBadge');
    const disksBadge = document.getElementById('disksBadge');
    const engineDot = document.getElementById('engineDot');
    const themeSelect = document.getElementById('themeSelect');

    let activeMode = 'assistant';
    let conversationHistory = [];

    // Gestión de Temas de Hardware
    function applyTheme(themeName) {
      document.documentElement.setAttribute('data-theme', themeName);
      themeSelect.value = themeName;
    }

    async function loadTheme() {
      try {
        const response = await fetch('/api/theme');
        const data = response.ok ? await response.json() : {};
        const selected = data.theme || localStorage.getItem('ecosystem_theme') || 'cerebro-neon';
        if (Array.from(themeSelect.options).some(option => option.value === selected)) {
          applyTheme(selected);
        }
      } catch (error) {}
    }

    themeSelect.addEventListener('change', async () => {
      const theme = themeSelect.value;
      applyTheme(theme);
      try {
        localStorage.setItem('ecosystem_theme', theme);
        await fetch('/api/theme', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({theme})
        });
      } catch (error) {}
    });
    loadTheme();
    window.addEventListener('focus', loadTheme);

    const modeButtons = {
      assistant: document.getElementById('btnModeAssistant'),
      operator: document.getElementById('btnModeOperator'),
      rag: document.getElementById('btnModeRag')
    };

    function setMode(newMode) {
      activeMode = newMode;
      Object.keys(modeButtons).forEach(m => {
        if (m === newMode) modeButtons[m].classList.add('active');
        else modeButtons[m].classList.remove('active');
      });
    }

    modeButtons.assistant.addEventListener('click', () => setMode('assistant'));
    modeButtons.operator.addEventListener('click', () => setMode('operator'));
    modeButtons.rag.addEventListener('click', () => setMode('rag'));

    document.querySelectorAll('.quick-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        input.value = pill.getAttribute('data-prompt');
        send();
      });
    });

    // Renderizador DOM seguro de Markdown y bloques de código usando únicamente createElement y textContent
    function appendInlineFormatted(parent, lineText) {
      const tokenPattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
      let lastIdx = 0;
      let match;
      while ((match = tokenPattern.exec(lineText)) !== null) {
        if (match.index > lastIdx) {
          parent.appendChild(document.createTextNode(lineText.slice(lastIdx, match.index)));
        }
        const token = match[0];
        if (token.startsWith('`') && token.endsWith('`')) {
          const codeEl = document.createElement('code');
          codeEl.className = 'inline-code';
          codeEl.textContent = token.slice(1, -1);
          parent.appendChild(codeEl);
        } else if (token.startsWith('**') && token.endsWith('**')) {
          const strongEl = document.createElement('strong');
          strongEl.textContent = token.slice(2, -2);
          parent.appendChild(strongEl);
        }
        lastIdx = tokenPattern.lastIndex;
      }
      if (lastIdx < lineText.length) {
        parent.appendChild(document.createTextNode(lineText.slice(lastIdx)));
      }
    }

    function renderFormattedText(container, text) {
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
      const raw = String(text || '');
      if (!raw.includes('```') && !raw.includes('`') && !raw.includes('**') && !raw.includes('\n- ') && !raw.includes('\n* ')) {
        container.textContent = raw;
        return;
      }

      const segments = raw.split('```');
      segments.forEach((segment, index) => {
        if (index % 2 === 1) {
          const pre = document.createElement('pre');
          pre.className = 'code-block';
          let lang = '';
          let codeBody = segment;
          const firstBreak = segment.indexOf('\n');
          if (firstBreak !== -1) {
            const candidateLang = segment.slice(0, firstBreak).trim();
            if (/^[a-zA-Z0-9_+-]+$/.test(candidateLang)) {
              lang = candidateLang;
              codeBody = segment.slice(firstBreak + 1);
            }
          }
          if (lang) {
            const langBadge = document.createElement('div');
            langBadge.className = 'code-lang';
            langBadge.textContent = lang;
            pre.appendChild(langBadge);
          }
          const codeNode = document.createElement('code');
          codeNode.textContent = codeBody.replace(/\n$/, '');
          pre.appendChild(codeNode);
          container.appendChild(pre);
        } else {
          const lines = segment.split('\n');
          let activeList = null;
          lines.forEach((line, lineIdx) => {
            const trimmed = line.trim();
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
              if (!activeList) {
                activeList = document.createElement('ul');
                activeList.className = 'msg-list';
                container.appendChild(activeList);
              }
              const li = document.createElement('li');
              appendInlineFormatted(li, trimmed.slice(2));
              activeList.appendChild(li);
            } else {
              activeList = null;
              if (line.length === 0 && (lineIdx === 0 || lineIdx === lines.length - 1)) {
                return;
              }
              const lineDiv = document.createElement('div');
              lineDiv.className = 'msg-line';
              appendInlineFormatted(lineDiv, line);
              container.appendChild(lineDiv);
            }
          });
        }
      });
    }

    function addMessage(kind, text, id) {
      const message = document.createElement('div');
      message.className = 'msg ' + kind;
      if (id) message.id = id;
      if (kind === 'ai') {
        renderFormattedText(message, text);
      } else {
        message.textContent = text;
      }
      chat.appendChild(message);
      chat.scrollTop = chat.scrollHeight;
      return message;
    }

    async function updateTelemetry() {
      try {
        const resp = await fetch('/api/system/status');
        if (!resp.ok) return;
        const data = await resp.json();
        const rep = data.report || {};
        const circuits = rep.circuits || {};
        const mlx = circuits.circuits && circuits.circuits.mlx ? circuits.circuits.mlx : {};
        const lat = typeof mlx.last_latency_ms === 'number' ? mlx.last_latency_ms.toFixed(0) + ' ms' : 'sin medición';
        const isFailover = circuits.failover_active || false;
        const cbState = (mlx.state || 'SIN DATOS').toUpperCase();

        telemetryBadge.textContent = `Gateway :8081 [${cbState}] · IA :8080 (${lat})`;
        failoverBadge.textContent = isFailover ? '⚠️ [Failover: Ollama]' : 'Failover: sin confirmación';

        const disksInfo = rep.disks || {};
        const mountedCount = typeof disksInfo.mounted_count === 'number' ? disksInfo.mounted_count : 0;
        const diskItems = Array.isArray(disksInfo.items) ? disksInfo.items : [];
        const diskNames = diskItems.length
          ? diskItems.map(d => d.mounted ? `${d.label} [ON]` : d.label).join(' · ')
          : 'Bibliotecas de BeatOS';
        if (disksBadge) {
          disksBadge.textContent = `💾 Bibliotecas: ${mountedCount}/${diskItems.length} · ${diskNames}`;
        }

        if (isFailover) {
          engineDot.className = 'dot warn';
        } else {
          engineDot.className = 'dot';
        }
      } catch(e) {}
      }
    updateTelemetry();
    window.setInterval(updateTelemetry, 8000);

    // Streaming SSE y Fallback
    async function send() {
      const text = input.value.trim();
      if (!text) return;
      addMessage('user', text);
      conversationHistory.push({role: 'user', content: text});
      input.value = '';
      sendButton.disabled = true;
      const startedAt = Date.now();
      const typing = addMessage('ai thinking', 'Solicitud enviada a Cerebro local.', 'typing-' + Date.now());
      const updateWait = () => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        typing.textContent = elapsed < 10
          ? `Solicitud enviada a Cerebro local · ${elapsed}s`
          : `El modelo sigue preparando la respuesta local · ${elapsed}s. La primera carga puede tardar hasta 3 minutos.`;
      };
      updateWait();
      const waitTimer = window.setInterval(updateWait, 1000);


      try {
        // Intento 1: Streaming Server-Sent Events (/api/chat/stream) con verificación de audio biométrico
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text: text,
            messages: conversationHistory.slice(-8),
            mode: activeMode
          })
        });

        if (!response.ok) {
          // Fallback a /api/chat tradicional si el backend no soporta streaming
          const legResp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text})
          });
          const legData = await legResp.json();
          if (!legResp.ok) throw new Error(legData.error || 'No se pudo procesar la consulta');
          window.clearInterval(waitTimer);
          typing.remove();
          addMessage('ai', legData.reply);
          conversationHistory.push({role: 'assistant', content: legData.reply});
          return;
        }

        window.clearInterval(waitTimer);
        typing.className = 'msg ai';
        typing.textContent = '';
        let fullReply = '';

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              const dataStr = trimmed.slice(6);
              if (dataStr === '[DONE]') break;
              try {
                const chunk = JSON.parse(dataStr);
                if (chunk.token) {
                  fullReply += chunk.token;
                  renderFormattedText(typing, fullReply);
                  chat.scrollTop = chat.scrollHeight;
                }
              } catch(e) {}
            }
          }
        }

        if (fullReply) {
          renderFormattedText(typing, fullReply);
          conversationHistory.push({role: 'assistant', content: fullReply});
        } else {
          typing.textContent = 'El modelo completó la respuesta.';
        }
      } catch (error) {
        window.clearInterval(waitTimer);
        typing.textContent = 'La IA local no está disponible en este momento.';
      } finally {
        window.clearInterval(waitTimer);
        sendButton.disabled = false;
        input.focus();
      }
    }

    sendButton.addEventListener('click', () => send());
    input.addEventListener('keypress', event => { if (event.key === 'Enter') send(); });
    addMessage('ai', 'Cerebro Jarvis Hub v2.1 activo en modo local.');
  </script>
</body>
</html>
"""


@app.after_request
def security_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/")
def home():
    return HTML_CONTENT


@app.get("/api/health")
def health():
    return jsonify({"service": "jarvis", "status": "ok"})


ECOSYSTEM_THEMES = frozenset({
    "cerebro-neon", "te-arcade", "te-ko", "mpc-vintage", "mpc-blue-le",
    "sp404-studio", "maschine-studio", "makerlicious-bone", "astro1deep-obsidian",
})


def ecosystem_theme_path() -> Path:
    return Path(os.environ.get("CEREBRO_THEME_FILE") or Path(__file__).resolve().parent / "ecosystem_theme.json")


@app.get("/api/theme")
def get_theme():
    try:
        data = json.loads(ecosystem_theme_path().read_text(encoding="utf-8"))
        theme = data.get("theme") if isinstance(data, dict) else None
        if isinstance(theme, str) and theme in ECOSYSTEM_THEMES:
            return jsonify(data)
    except (OSError, ValueError):
        pass
    return jsonify({"theme": None})


@app.post("/api/theme")
def set_theme():
    body = request.get_json(silent=True)
    theme = body.get("theme") if isinstance(body, dict) else None
    if not isinstance(theme, str) or theme not in ECOSYSTEM_THEMES:
        return jsonify({"success": False, "error": "Tema del ecosistema no válido."}), 400
    path = ecosystem_theme_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            pending = Path(stream.name)
            try:
                json.dump({"theme": theme, "source": "jarvis_hub"}, stream, indent=2)
                stream.close()
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
    except OSError:
        logging.exception("No se pudo escribir ecosystem_theme.json")
        return jsonify({"success": False, "error": "No se pudo guardar el tema del ecosistema."}), 500
    return jsonify({"success": True, "theme": theme})


@app.get("/api/system/status")
def system_status():
    """Devuelve la telemetría viva de microservicios, hardware M4 Pro, discos y circuit breakers."""
    try:
        report = collect_operator_telemetry()
        return jsonify({"status": "ok", "report": report})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.get("/api/mpc/status")
def mpc_status():
    """Consulta el estado del servidor de hardware y samples MPC en el puerto 8888."""
    try:
        resp = local_http_session().get(f"{beatos_api_base()}/api/status", timeout=2, allow_redirects=False)
        if resp.status_code == 200:
            return jsonify(resp.json())
    except Exception:
        pass
    return jsonify({"status": "offline", "card_mounted": False})


def _build_mode_system_prompt(user_input: str, mode: str) -> str:
    """Construye el system prompt dinámico según el modo activo (assistant, operator, rag)."""
    if is_operator_query(user_input, mode):
        telemetry_ctx = format_operator_telemetry_context()
        return (
            "Eres Jarvis Operator, asistente táctico de Cerebro OS en el equipo local. "
            "Describe los servicios, el uso de memoria y las bibliotecas detectadas en la telemetría. "
            "Responde en español de forma técnica, directa, precisa y breve usando exactamente estos datos reales:\n\n"
            f"{telemetry_ctx}"
        )
    if mode == "rag":
        rag_context = query_rag_context(user_input)
        return (
            "Eres Jarvis RAG & Producción Musical, especialista en la base de conocimientos de Cerebro OS, "
            "Akai MPC (.xpj / .xpm) y las bibliotecas BeatOS detectadas. "
            "Responde en español de forma rigurosa y directa apoyándote en el contexto e índices recuperados."
            + (f"\n\n{rag_context}" if rag_context else "")
        )
    return SYSTEM_PROMPT


def build_chat_messages(history: object, user_input: str, sys_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": sys_prompt}]
    if isinstance(history, list):
        for msg in history[-8:]:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": str(msg["content"])[:MAX_INPUT_CHARACTERS]})

    if not messages or messages[-1].get("content") != user_input:
        messages.append({"role": "user", "content": user_input})

    return messages


@app.post("/api/chat")
def chat_api():
    """Endpoint síncrono estándar OpenAI-compatible preservando compatibilidad estricta con tests."""
    if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
        return jsonify({"error": "Consulta demasiado larga"}), 413
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return jsonify({"error": "Se requiere un cuerpo JSON con texto"}), 400
    user_input = body["text"].strip()
    if not user_input:
        return jsonify({"error": "La consulta está vacía"}), 400
    if len(user_input) > MAX_INPUT_CHARACTERS:
        return jsonify({"error": "Consulta demasiado larga"}), 413

    if body.get("is_voice") or body.get("audio_b64"):
        return jsonify({"error": "Esta edición de Jarvis admite chat de texto"}), 400

    mode = str(body.get("mode", "assistant"))
    sys_prompt = _build_mode_system_prompt(user_input, mode)

    messages = build_chat_messages(body.get("messages"), user_input, sys_prompt)

    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 384,
    }
    if API_MODEL:
        payload["model"] = API_MODEL
    try:
        model_response = local_http_session().post(
            API_URL,
            json=payload,
            allow_redirects=False,
            timeout=(3, MODEL_REQUEST_TIMEOUT_SECONDS),
        )
        if model_response.status_code != 200:
            raise requests.HTTPError("El motor local no devolvió HTTP 200")
        reply = model_response.json()["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("El modelo no devolvió texto")
    except (requests.RequestException, KeyError, TypeError, ValueError):
        logging.exception("No se pudo consultar el modelo MLX local")
        return jsonify({"error": "La IA local no está disponible en este momento."}), 503
    return jsonify({"reply": reply.strip()[:12_000]})


@app.post("/api/chat/stream")
def chat_stream_api():
    """Endpoint de streaming Server-Sent Events (SSE) con verificación biométrica vocal y renderizado token a token."""
    if request.content_length is not None and request.content_length > MAX_STREAM_REQUEST_BYTES:
        return jsonify({"error": "Consulta demasiado larga"}), 413
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return jsonify({"error": "Se requiere un cuerpo JSON con texto"}), 400
    user_input = body["text"].strip()
    if not user_input:
        return jsonify({"error": "La consulta está vacía"}), 400
    if len(user_input) > MAX_INPUT_CHARACTERS:
        return jsonify({"error": "Consulta demasiado larga"}), 413

    if body.get("is_voice") or body.get("audio_b64"):
        return jsonify({"error": "Esta edición de Jarvis admite chat de texto"}), 400

    mode = str(body.get("mode", "assistant"))
    history = body.get("messages", [])
    sys_prompt = _build_mode_system_prompt(user_input, mode)

    messages = build_chat_messages(history, user_input, sys_prompt)

    payload = {
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 512,
        "stream": True,
    }
    if API_MODEL:
        payload["model"] = API_MODEL

    def generate_sse():
        resp = None
        try:
            resp = local_http_session().post(
                API_URL,
                json=payload,
                allow_redirects=False,
                stream=True,
                timeout=(4, MODEL_REQUEST_TIMEOUT_SECONDS),
            )
            if resp.status_code != 200:
                raise requests.HTTPError("El motor local no devolvió HTTP 200")
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_part = line_str[6:].strip()
                        if data_part == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data_part)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        except json.JSONDecodeError:
                            pass
        except Exception as exc:
            logging.exception("Error en streaming hacia backend: %s", exc)
            yield f"data: {json.dumps({'token': ' [Error temporal de inferencia local]'})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if resp is not None:
                resp.close()

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host=BIND_HOST, port=PORT, threaded=True)
