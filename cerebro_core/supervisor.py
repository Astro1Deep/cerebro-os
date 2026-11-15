"""Política de recuperación sin control directo de procesos."""

from __future__ import annotations


def recovery_decision(*, failures_in_window: int, max_failures: int) -> dict[str, object]:
    """Decide si una recuperación puede reintentarse dentro de un presupuesto fijo."""
    if max_failures < 1:
        raise ValueError("max_failures must be at least 1")
    if failures_in_window < 0:
        raise ValueError("failures_in_window cannot be negative")
    if failures_in_window >= max_failures:
        return {
            "state": "safe_mode",
            "restart_allowed": False,
            "reason": "failure_budget_exhausted",
        }
    return {
        "state": "recovery_available",
        "restart_allowed": True,
        "reason": "within_failure_budget",
    }
