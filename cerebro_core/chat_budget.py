"""Bounded conversation assembly for local models with finite KV memory."""

from __future__ import annotations

from typing import Iterable


CONTEXT_LABEL = "Contexto operativo local (curado, sin secretos):\n"


def _message(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("cada mensaje debe ser un objeto")
    role = value.get("role")
    content = value.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
        raise ValueError("el historial contiene un mensaje no válido")
    return {"role": role, "content": content.strip()}


def build_bounded_messages(
    *,
    system_prompt: str,
    curated_context: str,
    history: Iterable[dict[str, str]],
    max_characters: int = 64_000,
    context_characters: int = 10_000,
) -> list[dict[str, str]]:
    """Keep the latest complete turn within a conservative local context budget.

    Character budgeting is deliberately conservative for the Spanish local UI.
    It avoids presenting an unbounded history to an MLX server whose active KV
    window is finite.  The current user request is never discarded.
    """

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("el mensaje de sistema es obligatorio")
    if not 1_024 <= max_characters <= 128_000:
        raise ValueError("el presupuesto de contexto no es válido")
    context_characters = min(context_characters, max_characters)
    if context_characters < 0:
        raise ValueError("el límite de contexto no es válido")

    turns = [_message(item) for item in history]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("la consulta actual del usuario es obligatoria")

    system = system_prompt.strip()
    current = turns[-1]
    required = len(system) + len(current["content"])
    if required > max_characters:
        if required <= 128_000:
            max_characters = required
        else:
            raise ValueError("la consulta actual supera el presupuesto local")

    available_context = max(0, max_characters - required - len(CONTEXT_LABEL))
    context = curated_context.strip()[: min(context_characters, available_context)]
    result: list[dict[str, str]] = [{"role": "system", "content": system}]
    if context:
        result.append({"role": "system", "content": CONTEXT_LABEL + context})

    remaining = max_characters - sum(len(message["content"]) for message in result) - len(current["content"])
    selected: list[dict[str, str]] = [current]
    for turn in reversed(turns[:-1]):
        if len(turn["content"]) > remaining:
            break
        selected.insert(0, turn)
        remaining -= len(turn["content"])
    return [*result, *selected]
