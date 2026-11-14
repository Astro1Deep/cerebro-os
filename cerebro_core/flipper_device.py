"""Read-only local discovery for a physically attached Flipper workflow."""

from __future__ import annotations

from glob import glob
from pathlib import Path
import shutil
from typing import Final


SERIAL_GLOBS: Final[tuple[str, ...]] = (
    "/dev/cu.usbmodem*",
    "/dev/cu.usbserial*",
    "/dev/tty.usbmodem*",
    "/dev/tty.usbserial*",
)
KNOWN_TOOLS: Final[tuple[str, ...]] = ("qFlipper", "flipper", "fbt")


def discover_flipper_environment() -> dict[str, object]:
    """Report local attachment separately from management-tool availability.

    This deliberately never opens a serial port or sends commands.  A connected
    device is useful information even when qFlipper or a developer CLI is not
    installed, so callers must not conflate the two states.
    """

    serial_paths = sorted({path for pattern in SERIAL_GLOBS for path in glob(pattern) if Path(path).exists()})
    tools = {name: bool(shutil.which(name)) for name in KNOWN_TOOLS}
    device_attached = bool(serial_paths)
    management_tool_available = any(tools.values())
    return {
        "serial_candidates": serial_paths,
        "tools": tools,
        "device_attached": device_attached,
        "management_tool_available": management_tool_available,
        "ready_for_device_session": device_attached and management_tool_available,
        "mode": "discovery_only",
    }
