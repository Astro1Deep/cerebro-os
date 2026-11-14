"""Validation for local OpenAI-compatible model endpoints."""

from __future__ import annotations

import re
import urllib.request
from typing import Final
from urllib.parse import urlsplit


LOCAL_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
LOCAL_AI_PORTS: Final[frozenset[int]] = frozenset(range(8000, 12000))
CHAT_COMPLETIONS_PATH: Final[str] = "/v1/chat/completions"
DEFAULT_LOCAL_AI_ENDPOINT: Final[str] = (
    "http://127.0.0.1:8081/v1/chat/completions"
)
MODEL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def validate_loopback_endpoint(url: str, path: str) -> str:
    """Validate a direct loopback endpoint and its expected API path."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("La URL de IA local debe ser texto no vacío")

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("La URL de IA local no es válida") from error

    if (
        parsed.scheme != "http"
        or parsed.hostname not in LOCAL_HOSTS
        or port not in LOCAL_AI_PORTS
        or parsed.path != path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "La IA debe usar un endpoint OpenAI-compatible "
            "de bucle local autorizado"
        )

    return url.strip()


def validate_local_ai_endpoint(url: str) -> str:
    return validate_loopback_endpoint(url, CHAT_COMPLETIONS_PATH)


def configured_local_ai_endpoint(url: str | None = None) -> str:
    """Return a validated local endpoint, falling back only to the safe local default.

    Invalid, external or malformed configuration is never returned to the caller.
    This keeps prompts on the host even when a configuration value is unsafe.
    """

    candidate = (
        url.strip()
        if isinstance(url, str) and url.strip()
        else DEFAULT_LOCAL_AI_ENDPOINT
    )

    try:
        return validate_local_ai_endpoint(candidate)
    except ValueError:
        return DEFAULT_LOCAL_AI_ENDPOINT


def validate_local_ai_model(model: str) -> str:
    """Accept a conservative local model identifier without shell-like syntax."""

    if not isinstance(model, str) or not MODEL_NAME_PATTERN.fullmatch(model.strip()):
        raise ValueError("El modelo local no tiene un identificador válido")
    return model.strip()


def configured_local_ai_model(model: str | None = None) -> str | None:
    """Return a safe optional model name; an invalid value is never sent onward."""

    try:
        return validate_local_ai_model(model) if model else None
    except ValueError:
        return None


def models_endpoint(chat_endpoint: str) -> str:
    """Map a validated chat-completions URL to its local model-health endpoint."""

    validated = validate_local_ai_endpoint(chat_endpoint)
    return f"{validated[:-len(CHAT_COMPLETIONS_PATH)]}/v1/models"


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_http_open(request, *, timeout: float):
    """Open a direct local connection without proxy discovery or redirects."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirect()
    ).open(request, timeout=timeout)


def local_http_session():
    """Keep model data off environment-configured HTTP proxies."""
    import requests

    session = requests.Session()
    session.trust_env = False
    return session
