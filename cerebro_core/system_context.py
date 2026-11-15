"""Curated local-system facts supplied to the Cerebro chat client."""

from __future__ import annotations

from pathlib import Path


COMMERCIAL_MODULES = (
    "agente_investigador.py",
    "agente_lead_gen.py",
    "bot_saas_suscripcion.py",
    "pasarela_cobros_universal.py",
    "agente_orquestador_financiero.py",
    "gestor_billeteras_seguras.py",
)


def build_system_context(base_dir: Path | str, *, central_dir: Path | str | None = None) -> str:
    """Build a small verified map, never read credentials or arbitrary documents."""

    root = Path(base_dir).expanduser().resolve()
    central = Path(central_dir).expanduser().resolve() if central_dir else Path.home() / "Desktop" / "Cerebro_Central"
    commercial = central / "Agentes_Comerciales"
    existing = [name for name in COMMERCIAL_MODULES if (commercial / name).is_file()]
    facts = [
        "Contexto operativo local verificado de Cerebro.",
        f"Núcleo local: {root}.",
        "La inferencia principal usa un endpoint local de MLX; Ollama y Hermes son conectores opcionales.",
        "Existe un puente privado de Telegram autorizado por propietario/operadores y limitado a conversación, estado y diagnóstico; no ejecuta comandos remotos.",
        "Composer es un asistente musical local. Mayordomo crea propuestas de organización y exige una aprobación explícita para mover archivos.",
        f"Cerebro_Central: {'presente' if central.is_dir() else 'no disponible'} en {central}.",
    ]
    if existing:
        facts.extend(
            (
                f"El sistema comercial existe en {commercial}.",
                "Módulos comerciales detectados: " + ", ".join(existing) + ".",
                "Los módulos financieros, pagos o carteras son de alto riesgo: este chat puede analizarlos o proponer planes, pero no debe ejecutar pagos, operar carteras, ni publicar campañas sin aprobación humana explícita.",
            )
        )
    else:
        facts.append("No se han detectado módulos comerciales en la ruta conocida.")
    facts.extend(
        (
            "Este contexto no contiene secretos, tokens ni acceso libre a archivos. Si falta un dato, debe decirlo y proponer una comprobación local concreta, no afirmar que el sistema no existe.",
            "No hay navegación web automática: la adquisición web sólo se habilita por una orden explícita y las fuentes requieren aprobación antes de alimentar el RAG local.",
        )
    )
    return "\n".join(facts)
