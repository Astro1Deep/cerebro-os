"""Perfiles declarativos que separan contexto de autorización."""

from __future__ import annotations


PROFILES = (
    {
        "name": "studio",
        "purpose": "Producción local de audio y composición",
        "external_actions": "disabled",
    },
    {
        "name": "composer",
        "purpose": "Armonía, MIDI y dispositivos musicales",
        "external_actions": "disabled",
    },
    {
        "name": "control",
        "purpose": "Diagnóstico y propuestas de reparación",
        "external_actions": "approval_required",
    },
    {
        "name": "commercial",
        "purpose": "Investigación, análisis y borradores comerciales",
        "external_actions": "approval_required",
    },
    {
        "name": "developer",
        "purpose": "Desarrollo local con cambios revisables",
        "external_actions": "approval_required",
    },
)


def list_profiles() -> list[dict[str, str]]:
    return [dict(profile) for profile in PROFILES]
