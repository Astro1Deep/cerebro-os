"""Identidad visual local de Astro One Deep para interfaces de Cerebro."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO


ROOT_DIR = Path(__file__).resolve().parents[1]
TERMINAL_LOGO_PATH = ROOT_DIR / "branding" / "astro-one-deep-terminal.txt"
MOTION_FRAME_SECONDS = 0.08
MOTION_FRAMES = (
    (
        "                 .                          *",
        "       .                                      ",
        "---------------------+------------------------",
        "                 .-^^^-.                      ",
        "        .       /   O   \\        .             ",
        "================/_________\\===================",
        "     .        .       .        .               ",
    ),
    (
        "       .                          *           ",
        "                    .                         ",
        "------------------.-+-.-----------------------",
        "              .-'   |   '-.                   ",
        "       .     /      O      \\      .            ",
        "============/_______________\\================",
        "   .      .        .          .                ",
    ),
    (
        "   .                  *                       ",
        "         .                   .                ",
        "---------------.------+.------.---------------",
        "          .-'      /  |  \\      '-.            ",
        "     .   /        /   O   \\        \\     .     ",
        "========/________/_________\\________\\=========",
        " .      .          .          .                ",
    ),
    (
        "                  *                    .      ",
        "    .                         .               ",
        "---------------------+------------------------",
        "                 .-^^^-.                      ",
        "         .      /   O   \\        .             ",
        "================/_________\\===================",
        "     .     .        .        .                 ",
    ),
)
MOTION_FRAME_WIDTH = max(len(line) for frame in MOTION_FRAMES for line in frame)
MOTION_FRAME_HEIGHT = len(MOTION_FRAMES[0])
ANSI_VIOLET = "\x1b[95m"
ANSI_BLUE = "\x1b[94m"
ANSI_RESET = "\x1b[0m"


def render_welcome() -> str:
    """Render the optional terminal entry without ANSI escape sequences."""
    logo = TERMINAL_LOGO_PATH.read_text(encoding="utf-8").rstrip() if TERMINAL_LOGO_PATH.is_file() else "CEREBRO OS"
    return "\n".join(
        (
            "ASTRO ONE DEEP",
            "CEREBRO · LOCAL-FIRST",
            "",
            logo,
            "",
            "Usa `cerebro --help` para ver los comandos disponibles.",
        )
    )


def render_welcome_ansi() -> str:
    """Render the local welcome with basic ANSI colors and a closing reset."""
    logo = TERMINAL_LOGO_PATH.read_text(encoding="utf-8").rstrip() if TERMINAL_LOGO_PATH.is_file() else "CEREBRO OS"
    return "\n".join(
        (
            f"{ANSI_VIOLET}ASTRO ONE DEEP{ANSI_RESET}",
            f"{ANSI_BLUE}CEREBRO · LOCAL-FIRST{ANSI_RESET}",
            "",
            f"{ANSI_VIOLET}{logo}{ANSI_RESET}",
            "",
            f"Usa `cerebro --help` para ver los comandos disponibles.{ANSI_RESET}",
        )
    )


def terminal_supports_ansi(stream: TextIO) -> bool:
    """Return whether this stream is an interactive terminal with ANSI support enabled."""
    is_tty = getattr(stream, "isatty", lambda: False)
    return bool(is_tty()) and os.environ.get("TERM", "").lower() != "dumb"


def play_welcome_motion(stream: TextIO, *, sleep: Callable[[float], None] = time.sleep) -> bool:
    """Render a brief original terminal pulse only when explicitly requested in a TTY."""
    if not terminal_supports_ansi(stream):
        stream.write(f"{render_welcome()}\n")
        stream.flush()
        return False

    try:
        stream.write("\x1b[?25l")
        for index, frame in enumerate(MOTION_FRAMES):
            if index:
                stream.write(f"\x1b[{MOTION_FRAME_HEIGHT}A")
            stream.write("\n".join(line.ljust(MOTION_FRAME_WIDTH) for line in frame) + "\n")
            stream.flush()
            if index < len(MOTION_FRAMES) - 1:
                sleep(MOTION_FRAME_SECONDS)
        stream.write(f"\n{render_welcome()}\n")
    finally:
        stream.write("\x1b[?25h\n")
        stream.flush()
    return True
