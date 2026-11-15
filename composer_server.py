#!/usr/bin/env python3
"""Servidor local autónomo para el Asistente Musical de Cerebro.

Separado del REPL interactivo de mlx-cerebro-v3 para que la interfaz web
permanezca disponible aunque el terminal esté cerrado o Cerebro se reinicie.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from cerebro_core.local_endpoint import (
    configured_local_ai_endpoint,
    configured_local_ai_model,
    models_endpoint,
)

# Resolver recursos desde la carpeta del propio programa. Esto funciona tanto en
# ~/.mlx-stack (macOS) como en %USERPROFILE%\.cerebro-os (Windows) y permite
# instalar en una ruta personalizada sin editar el código.
STACK_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(STACK_DIR, "composer_ui.html")
COMPOSER_AUDIO_PATH = os.path.join(STACK_DIR, "composer_audio.js")
AI_URL = configured_local_ai_endpoint(
    os.getenv("CEREBRO_AI_URL", os.getenv("MLX_API_URL"))
)
AI_MODEL = configured_local_ai_model(os.getenv("CEREBRO_AI_MODEL"))
MODEL_REQUEST_TIMEOUT_SECONDS = 210
DEFAULT_LAN_TOKEN_FILE = os.path.join(STACK_DIR, ".composer_lan_token")
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-labelledby="title desc">
<title id="title">Cerebro Composer</title>
<desc id="desc">Teclado musical en color cian</desc>
<rect width="64" height="64" rx="14" fill="#090d12"/>
<rect x="10" y="13" width="44" height="38" rx="5" fill="#111923" stroke="#00d2ff" stroke-width="3"/>
<path d="M18 19v26M27 19v26M37 19v26M46 19v26M13 34h38" stroke="#eaf7ff" stroke-width="3" stroke-linecap="round"/>
<path d="M18 34v-9M37 34v-9" stroke="#ae9cff" stroke-width="5" stroke-linecap="round"/>
</svg>"""


def load_lan_token(token_path: str | None = None) -> str | None:
    """Carga la clave LAN desde un archivo privado; un archivo ausente deja Composer local."""
    selected_path = token_path or os.getenv("COMPOSER_LAN_TOKEN_FILE", DEFAULT_LAN_TOKEN_FILE)
    try:
        with open(selected_path, encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError:
        return None
    return token if len(token) >= 24 else None


def composer_bind_host(lan_token: str | None, requested_host: str | None = None) -> str:
    """Select the listener; a private reverse proxy may require loopback only."""
    if requested_host is not None:
        if requested_host != "127.0.0.1":
            raise ValueError("Composer solo admite 127.0.0.1 como enlace privado explícito")
        return requested_host
    return "0.0.0.0" if lan_token else "127.0.0.1"


LAN_CONNECT_PAGE = """<!doctype html>
<html lang="es"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conectando con Composer</title>
<body style="margin:0;padding:32px;background:#090d12;color:#eaf7ff;font:16px -apple-system,BlinkMacSystemFont,sans-serif">
<main><h1>Conectando con Composer…</h1><p id="status">Validando el acceso local.</p></main>
<script>
const status = document.getElementById("status");
const accessKey = new URLSearchParams(location.hash.slice(1)).get("access_key");
if (!accessKey) {
  status.textContent = "Falta la clave local en el enlace.";
} else {
  fetch("/api/lan-session", {method: "POST", headers: {"X-Composer-Access-Key": accessKey}})
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      history.replaceState(null, "", "/connect");
      location.replace("/");
    })
    .catch(() => { status.textContent = "No se pudo validar el acceso local."; });
}
</script></body></html>"""

NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}
PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_PATTERNS = {
    (0, 4, 7, 11): "mayor séptima",
    (0, 4, 7, 10): "séptima dominante",
    (0, 4, 8, 10): "séptima dominante aumentada",
    (0, 4, 6, 10): "séptima dominante ♭5",
    (0, 3, 7, 10): "menor séptima",
    (0, 3, 6, 10): "semidisminuido",
    (0, 3, 6, 9): "disminuido séptima",
    (0, 4, 7): "mayor",
    (0, 3, 7): "menor",
    (0, 3, 6): "disminuido",
    (0, 4, 8): "aumentado",
    (0, 2, 7): "sus2",
    (0, 5, 7): "sus4",
    (0, 2, 5, 7): "sus4(add9)",
}

SCALE_CATALOG = {
    "major": {"label": "Mayor (Jónica)", "family": "Diatónicas y modos", "intervals": (0, 2, 4, 5, 7, 9, 11)},
    "natural-minor": {"label": "Menor natural (Eólica)", "family": "Diatónicas y modos", "intervals": (0, 2, 3, 5, 7, 8, 10)},
    "dorian": {"label": "Dórica", "family": "Diatónicas y modos", "intervals": (0, 2, 3, 5, 7, 9, 10)},
    "phrygian": {"label": "Frigia", "family": "Diatónicas y modos", "intervals": (0, 1, 3, 5, 7, 8, 10)},
    "lydian": {"label": "Lidia", "family": "Diatónicas y modos", "intervals": (0, 2, 4, 6, 7, 9, 11)},
    "mixolydian": {"label": "Mixolidia", "family": "Diatónicas y modos", "intervals": (0, 2, 4, 5, 7, 9, 10)},
    "locrian": {"label": "Locria", "family": "Diatónicas y modos", "intervals": (0, 1, 3, 5, 6, 8, 10)},
    "harmonic-minor": {"label": "Menor armónica", "family": "Menores y modos avanzados", "intervals": (0, 2, 3, 5, 7, 8, 11)},
    "harmonic-major": {"label": "Mayor armónica", "family": "Menores y modos avanzados", "intervals": (0, 2, 4, 5, 7, 8, 11)},
    "dorian-sharp4": {"label": "Dórica #4", "family": "Menores y modos avanzados", "intervals": (0, 2, 3, 6, 7, 9, 10)},
    "phrygian-dominant": {"label": "Frigia dominante", "family": "Menores y modos avanzados", "intervals": (0, 1, 4, 5, 7, 8, 10)},
    "melodic-minor": {"label": "Menor melódica", "family": "Menores y modos avanzados", "intervals": (0, 2, 3, 5, 7, 9, 11)},
    "lydian-augmented": {"label": "Lidia aumentada", "family": "Menores y modos avanzados", "intervals": (0, 2, 4, 6, 8, 9, 11)},
    "lydian-dominant": {"label": "Lidia dominante", "family": "Menores y modos avanzados", "intervals": (0, 2, 4, 6, 7, 9, 10)},
    "super-locrian": {"label": "Superlocria (alterada)", "family": "Menores y modos avanzados", "intervals": (0, 1, 3, 4, 6, 8, 10)},
    "altered": {"label": "Alterada", "family": "Menores y modos avanzados", "intervals": (0, 1, 3, 4, 6, 8, 10)},
    "locrian-natural6": {"label": "Locria 6 natural", "family": "Modos de menor armónica", "intervals": (0, 1, 3, 5, 6, 9, 10)},
    "ionian-augmented": {"label": "Jónica aumentada", "family": "Modos de menor armónica", "intervals": (0, 2, 4, 5, 8, 9, 11)},
    "romanian-minor": {"label": "Menor rumana", "family": "Modos de menor armónica", "intervals": (0, 2, 3, 6, 7, 9, 10)},
    "lydian-sharp2": {"label": "Lidia #2", "family": "Modos de menor armónica", "intervals": (0, 3, 4, 6, 7, 9, 11)},
    "ultralocrian": {"label": "Ultralocria", "family": "Modos de menor armónica", "intervals": (0, 1, 3, 4, 6, 8, 9)},
    "whole-tone": {"label": "Tonos enteros", "family": "Simétricas y jazz", "intervals": (0, 2, 4, 6, 8, 10)},
    "diminished-half-whole": {"label": "Disminuida semitono-tono", "family": "Simétricas y jazz", "intervals": (0, 1, 3, 4, 6, 7, 9, 10)},
    "diminished-whole-half": {"label": "Disminuida tono-semitono", "family": "Simétricas y jazz", "intervals": (0, 2, 3, 5, 6, 8, 9, 11)},
    "augmented": {"label": "Aumentada", "family": "Simétricas y jazz", "intervals": (0, 3, 4, 7, 8, 11)},
    "bebop-major": {"label": "Bebop mayor", "family": "Simétricas y jazz", "intervals": (0, 2, 4, 5, 7, 8, 9, 11)},
    "bebop-dominant": {"label": "Bebop dominante", "family": "Simétricas y jazz", "intervals": (0, 2, 4, 5, 7, 9, 10, 11)},
    "bebop-dorian": {"label": "Bebop dórica", "family": "Simétricas y jazz", "intervals": (0, 2, 3, 4, 5, 7, 9, 10)},
    "chromatic": {"label": "Cromática (12 notas)", "family": "Simétricas y jazz", "intervals": tuple(range(12))},
    "major-pentatonic": {"label": "Pentatónica mayor", "family": "Pentatónicas y blues", "intervals": (0, 2, 4, 7, 9)},
    "minor-pentatonic": {"label": "Pentatónica menor", "family": "Pentatónicas y blues", "intervals": (0, 3, 5, 7, 10)},
    "blues": {"label": "Blues menor", "family": "Pentatónicas y blues", "intervals": (0, 3, 5, 6, 7, 10)},
    "major-blues": {"label": "Blues mayor", "family": "Pentatónicas y blues", "intervals": (0, 2, 3, 4, 7, 9)},
    "suspended-pentatonic": {"label": "Pentatónica suspendida", "family": "Pentatónicas y blues", "intervals": (0, 2, 5, 7, 10)},
    "hirajoshi": {"label": "Hirajoshi", "family": "Tradicionales y color", "intervals": (0, 2, 3, 7, 8)},
    "in-sen": {"label": "In Sen", "family": "Tradicionales y color", "intervals": (0, 1, 5, 7, 10)},
    "iwato": {"label": "Iwato", "family": "Tradicionales y color", "intervals": (0, 1, 5, 6, 10)},
    "kumoi": {"label": "Kumoi", "family": "Tradicionales y color", "intervals": (0, 2, 3, 7, 9)},
    "yo": {"label": "Yo", "family": "Tradicionales y color", "intervals": (0, 2, 5, 7, 9)},
    "spanish-8-tone": {"label": "Española de 8 tonos", "family": "Tradicionales y color", "intervals": (0, 1, 3, 4, 5, 7, 8, 10)},
    "bhairav": {"label": "Bhairav", "family": "Tradicionales y color", "intervals": (0, 1, 4, 5, 7, 8, 11)},
    "hungarian-minor": {"label": "Menor húngara", "family": "Tradicionales y color", "intervals": (0, 2, 3, 6, 7, 8, 11)},
    "double-harmonic": {"label": "Doble armónica", "family": "Tradicionales y color", "intervals": (0, 1, 4, 5, 7, 8, 11)},
    "neapolitan-minor": {"label": "Menor napolitana", "family": "Tradicionales y color", "intervals": (0, 1, 3, 5, 7, 8, 11)},
    "neapolitan-major": {"label": "Mayor napolitana", "family": "Tradicionales y color", "intervals": (0, 1, 3, 5, 7, 9, 11)},
    "persian": {"label": "Persa", "family": "Tradicionales y color", "intervals": (0, 1, 4, 5, 6, 8, 11)},
    "enigmatic": {"label": "Enigmática", "family": "Tradicionales y color", "intervals": (0, 1, 4, 6, 8, 10, 11)},
    "prometheus": {"label": "Prometeo", "family": "Tradicionales y color", "intervals": (0, 2, 4, 6, 9, 10)},
    "major-locrian": {"label": "Mayor locria", "family": "Tradicionales y color", "intervals": (0, 2, 4, 5, 6, 8, 10)},
}
SCALE_PATTERNS = {
    identifier: definition["intervals"] for identifier, definition in SCALE_CATALOG.items()
}
PUSH_MODES = {"in-key", "chromatic"}

Advisor = Callable[[list[str], str, dict[str, Any]], Awaitable[str]]


class ComposerRequest(BaseModel):
    notes: list[str] = Field(min_length=1, max_length=12)
    emotion: str = Field(default="Neutro, buscando progresiones naturales", max_length=500)
    tonic: str | None = Field(default=None, max_length=3)
    scale: str | None = Field(default=None, max_length=64)
    scale_intervals: str | None = Field(default=None, max_length=80)

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, notes: list[str]) -> list[str]:
        normalized: list[str] = []
        for note in notes:
            clean = note.strip().upper().replace("♯", "#").replace("♭", "B")
            if clean not in NOTE_TO_PC:
                raise ValueError(f"Nota no válida: {note}")
            canonical = PC_TO_NOTE[NOTE_TO_PC[clean]]
            if canonical not in normalized:
                normalized.append(canonical)
        if not normalized:
            raise ValueError("Selecciona al menos una nota")
        return normalized

    @field_validator("tonic")
    @classmethod
    def validate_tonic(cls, tonic: str | None) -> str | None:
        if tonic is None:
            return None
        clean = tonic.strip().upper().replace("♯", "#").replace("♭", "B")
        if clean not in NOTE_TO_PC:
            raise ValueError(f"Tónica no válida: {tonic}")
        return PC_TO_NOTE[NOTE_TO_PC[clean]]

    @model_validator(mode="after")
    def validate_scale_context(self):
        if self.tonic is None and self.scale is None:
            return self
        if not self.tonic or not self.scale:
            raise ValueError("La tónica y la escala deben enviarse juntas")
        resolve_scale_intervals(self.scale, self.scale_intervals)
        return self


class MidiNote(BaseModel):
    """Una nota concreta para exportación: pitch class, octava y dinámica MIDI."""

    note: str = Field(min_length=1, max_length=3)
    octave: int = Field(ge=0, le=8)
    velocity: int = Field(default=84, ge=1, le=127)

    @field_validator("note")
    @classmethod
    def validate_note(cls, note: str) -> str:
        clean = note.strip().upper().replace("♯", "#").replace("♭", "B")
        if clean not in NOTE_TO_PC:
            raise ValueError(f"Nota MIDI no válida: {note}")
        return PC_TO_NOTE[NOTE_TO_PC[clean]]


class MidiChordEvent(BaseModel):
    """Un acorde simultáneo y la duración que ocupa en pulsos."""

    notes: list[MidiNote] = Field(min_length=1, max_length=12)
    offset_beats: float = Field(default=0, ge=0, le=1024)
    duration_beats: float = Field(default=4, gt=0, le=1024)

    @model_validator(mode="after")
    def reject_duplicate_pitches(self):
        pitches = {
            (note.octave + 1) * 12 + NOTE_TO_PC[note.note] for note in self.notes
        }
        if len(pitches) != len(self.notes):
            raise ValueError("Un acorde MIDI no puede repetir la misma nota y octava")
        return self


class MidiExportRequest(BaseModel):
    """Cadena de acordes confirmada por la persona usuaria."""

    events: list[MidiChordEvent] = Field(min_length=1, max_length=128)
    tempo: int = Field(default=120, ge=30, le=300)
    beats_per_bar: int = Field(default=4, ge=1, le=12)
    beat_unit: int = Field(default=4)

    @field_validator("beat_unit")
    @classmethod
    def validate_beat_unit(cls, beat_unit: int) -> int:
        if beat_unit not in MIDI_BEAT_UNIT_POWERS:
            raise ValueError("La unidad de compás debe ser 1, 2, 4, 8 o 16")
        return beat_unit


class ProgressionRequest(BaseModel):
    """Petición explícita de una progresión completa desde un acorde elegido."""

    notes: list[MidiNote] = Field(min_length=2, max_length=12)
    tonic: str = Field(min_length=1, max_length=3)
    scale: str = Field(min_length=1, max_length=64)
    scale_intervals: str | None = Field(default=None, max_length=80)
    style: Literal[
        "adaptive-link",
        "diatonic",
        "cadence",
        "secondary-dominant",
        "diminished-passing",
        "augmented-dominant",
        "diminished-link",
    ] = "adaptive-link"
    length: int = Field(default=4, ge=2, le=8)
    duration_beats: float = Field(default=4, gt=0, le=64)
    allow_unresolved_source: bool = False

    @field_validator("tonic")
    @classmethod
    def validate_tonic(cls, tonic: str) -> str:
        clean = tonic.strip().upper().replace("♯", "#").replace("♭", "B")
        if clean not in NOTE_TO_PC:
            raise ValueError(f"Tónica no válida: {tonic}")
        return PC_TO_NOTE[NOTE_TO_PC[clean]]

    @model_validator(mode="after")
    def validate_scale(self):
        resolve_scale_intervals(self.scale, self.scale_intervals)
        return self


class HealthResponse(BaseModel):
    status: str
    ai: str
    ui: str


MIDI_TICKS_PER_BEAT = 480
MIDI_BEAT_UNIT_POWERS = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
MAX_MIDI_IMPORT_BYTES = 2 * 1024 * 1024
MAX_MIDI_IMPORT_EVENTS = 4096


def encode_midi_variable_length(value: int) -> bytes:
    """Codifica un delta-time MIDI de longitud variable."""
    if value < 0:
        raise ValueError("El delta-time MIDI no puede ser negativo")
    encoded = [value & 0x7F]
    value >>= 7
    while value:
        encoded.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(encoded))


def midi_pitch(note: MidiNote) -> int:
    pitch = (note.octave + 1) * 12 + NOTE_TO_PC[note.note]
    if not 0 <= pitch <= 127:
        raise ValueError("La nota MIDI está fuera del rango 0–127")
    return pitch


def build_midi_file(export: MidiExportRequest) -> bytes:
    """Construye un Standard MIDI File de una pista sin dependencias externas."""
    track_name = "Cerebro Composer".encode("ascii")
    tempo_microseconds = 60_000_000 // export.tempo
    track = bytearray()
    track.extend(b"\x00\xff\x03")
    track.extend(encode_midi_variable_length(len(track_name)))
    track.extend(track_name)
    track.extend(b"\x00\xff\x51\x03")
    track.extend(tempo_microseconds.to_bytes(3, "big"))
    track.extend(
        bytes(
            (
                0,
                0xFF,
                0x58,
                0x04,
                export.beats_per_bar,
                MIDI_BEAT_UNIT_POWERS[export.beat_unit],
                24,
                8,
            )
        )
    )

    for event in export.events:
        chord = sorted((midi_pitch(note), note.velocity) for note in event.notes)
        offset = round(event.offset_beats * MIDI_TICKS_PER_BEAT)
        for index, (pitch, velocity) in enumerate(chord):
            track.extend(encode_midi_variable_length(offset if index == 0 else 0))
            track.extend(bytes((0x90, pitch, velocity)))

        duration = max(1, round(event.duration_beats * MIDI_TICKS_PER_BEAT))
        for index, (pitch, _) in enumerate(chord):
            track.extend(encode_midi_variable_length(duration if index == 0 else 0))
            track.extend(bytes((0x80, pitch, 0)))

    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd" + (6).to_bytes(4, "big") + bytes((0, 0, 0, 1)) + MIDI_TICKS_PER_BEAT.to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def read_midi_variable_length(data: bytes, position: int, end: int) -> tuple[int, int]:
    """Lee un delta-time o longitud SMF con un límite defensivo de cuatro bytes."""
    value = 0
    for _ in range(4):
        if position >= end:
            raise ValueError("MIDI truncado al leer una longitud variable")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, position
    raise ValueError("MIDI inválido: longitud variable demasiado larga")


def parse_midi_track(track: bytes, track_number: int) -> list[dict[str, Any]]:
    """Extrae eventos de notas y metadatos de una pista SMF sin ejecutar contenido externo."""
    events: list[dict[str, Any]] = []
    position = 0
    tick = 0
    event_order = 0
    running_status: int | None = None
    track_end = len(track)

    while position < track_end:
        delta, position = read_midi_variable_length(track, position, track_end)
        tick += delta
        if position >= track_end:
            raise ValueError("MIDI truncado después de un delta-time")

        first_byte = track[position]
        if first_byte & 0x80:
            status = first_byte
            position += 1
            running_status = status if 0x80 <= status <= 0xEF else None
        else:
            if running_status is None:
                raise ValueError("MIDI inválido: falta el estado de un evento")
            status = running_status

        if status == 0xFF:
            if position >= track_end:
                raise ValueError("MIDI truncado en un evento meta")
            meta_type = track[position]
            position += 1
            length, position = read_midi_variable_length(track, position, track_end)
            if position + length > track_end:
                raise ValueError("MIDI truncado en datos meta")
            payload = track[position : position + length]
            position += length
            if meta_type == 0x51 and length == 3:
                events.append(
                    {
                        "kind": "tempo",
                        "tick": tick,
                        "microseconds": int.from_bytes(payload, "big"),
                        "track": track_number,
                        "order": event_order,
                    }
                )
            elif meta_type == 0x58 and length >= 2:
                events.append(
                    {
                        "kind": "signature",
                        "tick": tick,
                        "numerator": payload[0],
                        "denominator_power": payload[1],
                        "track": track_number,
                        "order": event_order,
                    }
                )
            event_order += 1
            continue

        if status in {0xF0, 0xF7}:
            length, position = read_midi_variable_length(track, position, track_end)
            if position + length > track_end:
                raise ValueError("MIDI truncado en un evento SysEx")
            position += length
            event_order += 1
            continue

        if not 0x80 <= status <= 0xEF:
            system_lengths = {0xF1: 1, 0xF2: 2, 0xF3: 1, 0xF6: 0, 0xF8: 0, 0xFA: 0, 0xFB: 0, 0xFC: 0, 0xFE: 0}
            if status not in system_lengths:
                raise ValueError("MIDI contiene un evento de sistema no compatible")
            length = system_lengths[status]
            if position + length > track_end:
                raise ValueError("MIDI truncado en un evento de sistema")
            position += length
            event_order += 1
            continue

        event_type = status & 0xF0
        data_length = 1 if event_type in {0xC0, 0xD0} else 2
        if position + data_length > track_end:
            raise ValueError("MIDI truncado en un evento de canal")
        data_bytes = track[position : position + data_length]
        position += data_length
        if event_type in {0x80, 0x90}:
            pitch, velocity = data_bytes
            events.append(
                {
                    "kind": "on" if event_type == 0x90 and velocity else "off",
                    "tick": tick,
                    "channel": status & 0x0F,
                    "pitch": pitch,
                    "velocity": velocity,
                    "track": track_number,
                    "order": event_order,
                }
            )
        event_order += 1

    return events


def parse_midi_file(data: bytes) -> dict[str, Any]:
    """Importa un SMF 0/1 de notas a la cadena editable de Composer.

    La interfaz trabaja con bloques de notas; por ello conserva archivos de
    acordes y melodías simples y reporta cualquier solapamiento no representable.
    """
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("El archivo no es un MIDI estándar válido")
    header_length = int.from_bytes(data[4:8], "big")
    if header_length < 6 or 8 + header_length > len(data):
        raise ValueError("La cabecera MIDI es inválida o está truncada")

    midi_format = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    ticks_per_beat = int.from_bytes(data[12:14], "big")
    if midi_format not in {0, 1}:
        raise ValueError("Composer admite MIDI estándar de formato 0 o 1")
    if track_count < 1:
        raise ValueError("El MIDI no contiene pistas")
    if ticks_per_beat & 0x8000 or ticks_per_beat == 0:
        raise ValueError("Composer no admite MIDI con división SMPTE")

    position = 8 + header_length
    parsed_events: list[dict[str, Any]] = []
    for track_number in range(track_count):
        if position + 8 > len(data) or data[position : position + 4] != b"MTrk":
            raise ValueError("MIDI inválido: falta una pista")
        track_length = int.from_bytes(data[position + 4 : position + 8], "big")
        track_start = position + 8
        track_end = track_start + track_length
        if track_end > len(data):
            raise ValueError("MIDI truncado dentro de una pista")
        parsed_events.extend(parse_midi_track(data[track_start:track_end], track_number))
        if len(parsed_events) > MAX_MIDI_IMPORT_EVENTS * 8:
            raise ValueError("El MIDI contiene demasiados eventos para Composer")
        position = track_end

    tempo_events = sorted(
        (event for event in parsed_events if event["kind"] == "tempo"),
        key=lambda event: (event["tick"], event["track"], event["order"]),
    )
    tempo = 120
    if tempo_events and tempo_events[0]["microseconds"]:
        tempo = round(60_000_000 / tempo_events[0]["microseconds"])
    tempo = min(300, max(30, tempo))

    warnings: list[str] = []
    signature_events = sorted(
        (event for event in parsed_events if event["kind"] == "signature"),
        key=lambda event: (event["tick"], event["track"], event["order"]),
    )
    beats_per_bar, beat_unit = 4, 4
    if signature_events:
        signature = signature_events[0]
        candidate_unit = 2 ** signature["denominator_power"]
        if 1 <= signature["numerator"] <= 12 and candidate_unit in MIDI_BEAT_UNIT_POWERS:
            beats_per_bar = signature["numerator"]
            beat_unit = candidate_unit
        else:
            warnings.append("El compás del MIDI no es compatible; se usará 4/4.")

    note_events = sorted(
        (event for event in parsed_events if event["kind"] in {"on", "off"}),
        key=lambda event: (
            event["tick"],
            0 if event["kind"] == "off" else 1,
            event["track"],
            event["order"],
        ),
    )
    active_notes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    completed_notes: list[dict[str, int]] = []
    for event in note_events:
        note_key = (event["channel"], event["pitch"])
        if event["kind"] == "on":
            active_notes[note_key].append(event)
            continue
        if not active_notes[note_key]:
            continue
        start = active_notes[note_key].pop(0)
        completed_notes.append(
            {
                "start": start["tick"],
                "duration": max(1, event["tick"] - start["tick"]),
                "pitch": start["pitch"],
                "velocity": start["velocity"],
            }
        )

    for pending in active_notes.values():
        for start in pending:
            warnings.append("Había notas sin Note Off; se les asignó una duración de un pulso.")
            completed_notes.append(
                {
                    "start": start["tick"],
                    "duration": ticks_per_beat,
                    "pitch": start["pitch"],
                    "velocity": start["velocity"],
                }
            )

    if not completed_notes:
        raise ValueError("El MIDI no contiene notas completas para importar")

    grouped_notes: dict[int, list[dict[str, int]]] = defaultdict(list)
    for event in completed_notes:
        grouped_notes[event["start"]].append(event)
    if len(grouped_notes) > MAX_MIDI_IMPORT_EVENTS:
        raise ValueError("El MIDI contiene demasiados bloques de notas para Composer")

    imported_events: list[dict[str, Any]] = []
    cursor_tick = 0
    for start_tick in sorted(grouped_notes):
        chord_notes = sorted(grouped_notes[start_tick], key=lambda note: note["pitch"])
        durations = {note["duration"] for note in chord_notes}
        duration_ticks = min(durations)
        if len(durations) > 1:
            warnings.append("Un bloque tenía duraciones distintas; Composer conserva la más corta para mantener el acorde editable.")
        if start_tick < cursor_tick:
            warnings.append("Había notas solapadas; Composer las convirtió en bloques consecutivos editables.")
        offset_ticks = max(0, start_tick - cursor_tick)
        cursor_tick = max(cursor_tick, start_tick + duration_ticks)
        imported_events.append(
            {
                "notes": [
                    {
                        "note": PC_TO_NOTE[note["pitch"] % 12],
                        "octave": note["pitch"] // 12 - 1,
                        "velocity": note["velocity"],
                    }
                    for note in chord_notes
                ],
                "offset_beats": round(offset_ticks / ticks_per_beat, 6),
                "duration_beats": round(duration_ticks / ticks_per_beat, 6),
            }
        )

    return {
        "tempo": tempo,
        "beats_per_bar": beats_per_bar,
        "beat_unit": beat_unit,
        "events": imported_events,
        "warnings": list(dict.fromkeys(warnings)),
    }


def find_partial_chord_options(
    pcs: list[int], scale_pcs: set[int] | None = None
) -> list[tuple[int, str, list[str]]]:
    """Propone sólo los cierres más pequeños compatibles con el conjunto y escala."""
    candidates: list[tuple[int, int, str, list[str]]] = []

    for root_pc in pcs:
        selected_intervals = {(pc - root_pc) % 12 for pc in pcs}
        for pattern, quality in CHORD_PATTERNS.items():
            pattern_set = set(pattern)
            if not selected_intervals < pattern_set:
                continue

            chord_pcs = {(root_pc + interval) % 12 for interval in pattern_set}
            if scale_pcs is not None and not chord_pcs.issubset(scale_pcs):
                continue

            missing_notes = [
                PC_TO_NOTE[(root_pc + interval) % 12]
                for interval in sorted(pattern_set - selected_intervals)
            ]
            candidates.append((len(missing_notes), root_pc, quality, missing_notes))

    if not candidates:
        return []

    fewest_missing = min(candidate[0] for candidate in candidates)
    best_candidates = sorted(
        (candidate for candidate in candidates if candidate[0] == fewest_missing),
        key=lambda candidate: (candidate[1], candidate[2], candidate[3]),
    )
    return [
        (root_pc, quality, missing_notes)
        for _, root_pc, quality, missing_notes in best_candidates
    ]


def find_unique_partial_chord(
    pcs: list[int], scale_pcs: set[int] | None = None
) -> tuple[int, str, list[str]] | None:
    """Completa un acorde sólo cuando una única opción es compatible con la escala."""
    options = find_partial_chord_options(pcs, scale_pcs)
    if len(options) != 1:
        return None

    return options[0]


def unresolved_transition_options(tonic: str | None) -> list[dict[str, Any]]:
    """Rutas funcionales disponibles para un cluster que no define un acorde único."""
    if tonic is None:
        return []
    return [
        {
            "id": "cadence",
            "style": "cadence",
            "length": 4,
            "label": "Cadencia clara",
            "path": "IV → V7 → I",
        },
        {
            "id": "diminished-passing",
            "style": "diminished-passing",
            "length": 4,
            "label": "Tensión con disminuido",
            "path": "♯IV°7 → V7 → I",
        },
        {
            "id": "augmented-dominant",
            "style": "augmented-dominant",
            "length": 4,
            "label": "Tensión con dominante aumentada",
            "path": "ii → V+7 → I",
        },
        {
            "id": "secondary-dominant",
            "style": "secondary-dominant",
            "length": 5,
            "label": "Dominante secundaria",
            "path": "V/ii → ii → V7 → I",
        },
    ]


def analyze_notes(
    notes: list[str], tonic: str | None = None, scale_intervals: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """Identifica acordes completos o parciales y sugiere progresiones sin usar IA."""
    pcs = sorted({NOTE_TO_PC[note.strip().upper()] for note in notes})
    chord = "Conjunto de notas: " + ", ".join(PC_TO_NOTE[pc] for pc in pcs)
    root_pc = pcs[0]
    quality = None
    missing_notes: list[str] = []
    completion_options: list[dict[str, Any]] = []

    # Probar cada nota como posible fundamental y preferir patrones completos.
    for candidate in pcs:
        intervals = tuple(sorted((pc - candidate) % 12 for pc in pcs))
        if intervals in CHORD_PATTERNS:
            root_pc = candidate
            quality = CHORD_PATTERNS[intervals]
            chord = f"{PC_TO_NOTE[root_pc]} {quality}"
            break

    scale_pcs: set[int] | None = None
    if tonic is not None and scale_intervals is not None:
        canonical_tonic = tonic.strip().upper().replace("♯", "#").replace("♭", "B")
        if canonical_tonic in NOTE_TO_PC:
            tonic_pc = NOTE_TO_PC[canonical_tonic]
            scale_pcs = {(tonic_pc + interval) % 12 for interval in scale_intervals}

    if quality is None:
        partial_options = find_partial_chord_options(pcs, scale_pcs)
        completion_options = [
            {
                "root": PC_TO_NOTE[candidate_root],
                "quality": candidate_quality,
                "chord": f"{PC_TO_NOTE[candidate_root]} {candidate_quality}",
                "missing_notes": candidate_missing_notes,
            }
            for candidate_root, candidate_quality, candidate_missing_notes in partial_options
        ]
        partial_chord = find_unique_partial_chord(pcs, scale_pcs)
        if partial_chord is not None:
            root_pc, quality, missing_notes = partial_chord
            chord = f"{PC_TO_NOTE[root_pc]} {quality} (incompleto)"

    if quality is None:
        return {
            "chord": chord,
            "root": None,
            "quality": "no determinada",
            "scales": [],
            "next_roots": [],
            "missing_notes": [],
            "completion_options": completion_options,
            "transition_options": unresolved_transition_options(tonic),
        }

    root = PC_TO_NOTE[root_pc]
    relative_minor = PC_TO_NOTE[(root_pc + 9) % 12]
    relative_major = PC_TO_NOTE[(root_pc + 3) % 12]

    if quality and "menor" in quality:
        scales = [f"{root} menor natural", f"{root} menor armónica", f"{relative_major} mayor"]
    elif quality == "séptima dominante":
        scales = [f"{root} mixolidia", f"{root} blues", f"{PC_TO_NOTE[(root_pc + 5) % 12]} mayor"]
    elif quality == "séptima dominante ♭5":
        scales = [f"{root} lidia dominante", f"{root} tonos enteros", f"{root} mixolidia ♯4"]
    elif quality == "disminuido" or quality == "semidisminuido":
        scales = [f"{root} disminuida", f"{root} locria", f"{PC_TO_NOTE[(root_pc + 1) % 12]} menor armónica"]
    else:
        scales = [f"{root} mayor", f"{relative_minor} menor", f"{root} pentatónica mayor"]

    next_roots = [
        PC_TO_NOTE[(root_pc + 5) % 12],  # IV
        PC_TO_NOTE[(root_pc + 7) % 12],  # V
        PC_TO_NOTE[(root_pc + 9) % 12],  # vi
    ]
    return {
        "chord": chord,
        "root": root,
        "quality": quality or "no determinada",
        "scales": scales,
        "next_roots": next_roots,
        "missing_notes": missing_notes,
        "completion_options": completion_options,
        "transition_options": [],
    }


def scale_chord_definition(
    tonic_pc: int, scale_intervals: tuple[int, ...], degree: int, seventh: bool = False
) -> tuple[int, tuple[int, ...], str]:
    """Construye un acorde por grados para una escala concreta."""
    note_count = 4 if seventh else 3
    scale_size = len(scale_intervals)
    root_pc = (tonic_pc + scale_intervals[degree % scale_size]) % 12
    pitch_classes = [
        (tonic_pc + scale_intervals[(degree + step * 2) % scale_size]) % 12
        for step in range(note_count)
    ]
    intervals = tuple(sorted((pitch - root_pc) % 12 for pitch in pitch_classes))
    quality = CHORD_PATTERNS.get(intervals, "acorde de escala")
    return root_pc, intervals, quality


def voice_generated_chord(
    root_pc: int, intervals: tuple[int, ...], reference_notes: list[MidiNote]
) -> list[dict[str, int | str]]:
    """Da una posición cercana al acorde previo, manteniendo una conducción simple."""
    pitches = [48 + root_pc + interval for interval in sorted(intervals)]
    reference_pitches = [midi_pitch(note) for note in reference_notes]
    reference_center = sum(reference_pitches) / len(reference_pitches)
    while sum(pitches) / len(pitches) < reference_center - 5:
        pitches = [pitch + 12 for pitch in pitches]
    while sum(pitches) / len(pitches) > reference_center + 7:
        pitches = [pitch - 12 for pitch in pitches]
    velocity = round(sum(note.velocity for note in reference_notes) / len(reference_notes))
    return [
        {
            "note": PC_TO_NOTE[pitch % 12],
            "octave": pitch // 12 - 1,
            "velocity": velocity,
        }
        for pitch in pitches
    ]


def closing_steps(
    lead_in: list[tuple[int, tuple[int, ...], str]],
    tonic_chord: tuple[int, tuple[int, ...], str],
    count: int,
) -> list[tuple[int, tuple[int, ...], str]]:
    """Devuelve ``count`` acordes que terminan siempre en la tónica.

    Los enlaces más inmediatos se sitúan al final de ``lead_in``. Al pedir una
    cadena corta se conservan precisamente esos acordes, sin dejar un V o un
    acorde de paso sin resolver.
    """
    if count <= 0:
        return []
    if count == 1 or not lead_in:
        return [tonic_chord] * count

    required_lead_in = count - 1
    repetitions = (required_lead_in + len(lead_in) - 1) // len(lead_in)
    expanded_lead_in = (lead_in * repetitions)[-required_lead_in:]
    return [*expanded_lead_in, tonic_chord]


def build_progression(request: ProgressionRequest) -> dict[str, Any]:
    """Crea una cadena MIDI explícita: origen, enlaces y acordes de resolución."""
    source_pitch_classes = list(dict.fromkeys(note.note for note in request.notes))
    scale_intervals = resolve_scale_intervals(request.scale, request.scale_intervals)
    theory = analyze_notes(
        source_pitch_classes,
        tonic=request.tonic,
        scale_intervals=scale_intervals,
    )
    source_is_incomplete = "(incompleto)" in theory["chord"]
    source_is_unresolved = theory["quality"] == "no determinada"
    if source_is_incomplete or (source_is_unresolved and not request.allow_unresolved_source):
        raise ValueError("Selecciona un acorde completo antes de construir una progresión MIDI")
    if source_is_unresolved and request.style == "diatonic":
        raise ValueError("Un conjunto sin acorde definido necesita una ruta de transición explícita")

    tonic_pc = NOTE_TO_PC[request.tonic]
    scale_pcs = [(tonic_pc + interval) % 12 for interval in scale_intervals]
    source_root_pc = NOTE_TO_PC[theory["root"]] if theory["root"] else None
    source_degree = scale_pcs.index(source_root_pc) if source_root_pc in scale_pcs else None
    if request.style == "diatonic" and source_degree is None:
        raise ValueError("La fundamental del acorde de origen no pertenece a la escala seleccionada")

    source_event = {
        "label": f"{theory['chord']} · {'color de origen' if source_is_unresolved else 'origen'}",
        "notes": [
            {"note": note.note, "octave": note.octave, "velocity": note.velocity}
            for note in request.notes
        ],
        "offset_beats": 0,
        "duration_beats": request.duration_beats,
    }
    events: list[dict[str, Any]] = [source_event]
    remaining = request.length - 1

    if request.style == "diatonic":
        degree_paths = {
            0: [(5, False), (1, False), (4, True), (0, False)],
            1: [(4, True), (0, False), (5, False), (3, False)],
            2: [(5, False), (1, False), (4, True), (0, False)],
            3: [(1, False), (4, True), (0, False), (5, False)],
            4: [(0, False), (5, False), (1, False), (3, False)],
            5: [(1, False), (4, True), (0, False), (3, False)],
            6: [(0, False), (5, False), (1, False), (4, True)],
        }
        steps: list[tuple[int, tuple[int, ...], str]] = []
        for degree, use_seventh in degree_paths[source_degree % 7]:
            root_pc, intervals, quality = scale_chord_definition(
                tonic_pc, scale_intervals, degree, use_seventh
            )
            steps.append((root_pc, intervals, quality))
    else:
        tonic_chord = scale_chord_definition(tonic_pc, scale_intervals, 0)
        supertonic_chord = scale_chord_definition(tonic_pc, scale_intervals, 1)
        subdominant_chord = scale_chord_definition(tonic_pc, scale_intervals, 3)
        dominant_chord = (
            (tonic_pc + 7) % 12,
            (0, 4, 7, 10),
            "séptima dominante",
        )
        augmented_dominant_chord = (
            (tonic_pc + 7) % 12,
            (0, 4, 8, 10),
            "séptima dominante aumentada",
        )
        diminished_passing_chord = (
            (tonic_pc + 6) % 12,
            (0, 3, 6, 9),
            "disminuido séptima de paso",
        )
        secondary_dominant_chord = (
            (supertonic_chord[0] + 7) % 12,
            (0, 4, 7, 10),
            "séptima dominante secundaria (V/ii)",
        )

        if request.style == "cadence":
            lead_in = [tonic_chord, subdominant_chord, dominant_chord]
        elif request.style in {"diminished-link", "diminished-passing"}:
            lead_in = [tonic_chord, diminished_passing_chord, dominant_chord]
        elif request.style == "augmented-dominant":
            lead_in = [tonic_chord, supertonic_chord, augmented_dominant_chord]
        elif request.style == "secondary-dominant":
            lead_in = [
                tonic_chord,
                secondary_dominant_chord,
                supertonic_chord,
                dominant_chord,
            ]
        else:  # adaptive-link: evita inventar una única receta para todos los acordes.
            source_quality = theory["quality"]
            if source_root_pc == dominant_chord[0] and "séptima dominante" in source_quality:
                lead_in = [tonic_chord, dominant_chord]
            elif "aumentado" in source_quality:
                lead_in = [tonic_chord, augmented_dominant_chord]
            elif "disminuido" in source_quality:
                lead_in = [tonic_chord, dominant_chord]
            elif source_degree is None:
                lead_in = [
                    tonic_chord,
                    secondary_dominant_chord,
                    supertonic_chord,
                    dominant_chord,
                ]
            else:
                lead_in = [tonic_chord, subdominant_chord, dominant_chord]
        steps = closing_steps(lead_in, tonic_chord, remaining)

    reference_notes = request.notes
    for index in range(remaining):
        root_pc, intervals, quality = steps[index % len(steps)]
        generated_notes = voice_generated_chord(root_pc, intervals, reference_notes)
        events.append(
            {
                "label": f"{PC_TO_NOTE[root_pc]} {quality}",
                "notes": generated_notes,
                "offset_beats": 0,
                "duration_beats": request.duration_beats,
            }
        )
        reference_notes = [MidiNote(**note) for note in generated_notes]

    return {
        "events": events,
        "description": " → ".join(event["label"].replace(" · origen", "") for event in events),
        "theory": theory,
    }


def format_theory(theory: dict[str, Any]) -> str:
    scales = ", ".join(theory["scales"]) or "sin inferir; elige una tónica y escala"
    next_roots = ", ".join(theory["next_roots"]) or "sin sugerencia automática"
    summary = (
        f"Acorde detectado: {theory['chord']}\n"
        f"Escalas compatibles: {scales}\n"
        f"Siguientes fundamentales sugeridas: {next_roots}"
    )
    context = theory.get("performance_context")
    if context:
        summary += f"\nContexto de interpretación: {context['label']}"
    return summary


def resolve_scale_intervals(scale: str, custom_intervals: str | None = None) -> tuple[int, ...]:
    if scale != "custom":
        if scale not in SCALE_PATTERNS:
            raise ValueError("Escala no válida")
        return SCALE_PATTERNS[scale]

    try:
        values = tuple(sorted({int(value.strip()) for value in (custom_intervals or "").split(",") if value.strip()}))
    except ValueError as exc:
        raise ValueError("Los intervalos deben ser enteros entre 0 y 11") from exc
    if not values or values[0] != 0 or any(value < 0 or value > 11 for value in values):
        raise ValueError("La escala personalizada debe incluir 0 y usar intervalos entre 0 y 11")
    return values


def build_push_grid(
    root: str, scale_intervals: tuple[int, ...], mode: str, octave: int
) -> list[list[dict[str, Any]]]:
    """Returna una rejilla 8x8 en la disposición interválica de Push."""
    root_pc = NOTE_TO_PC[root]
    grid: list[list[dict[str, Any]]] = []

    for row in range(8):
        rendered_row: list[dict[str, Any]] = []
        row_offset = (7 - row) * (3 if mode == "in-key" else 5)
        for column in range(8):
            offset = row_offset + column
            if mode == "in-key":
                octave_offset, degree = divmod(offset, len(scale_intervals))
                raw_pitch = root_pc + scale_intervals[degree]
                pad_octave = octave + octave_offset + raw_pitch // 12
                is_root = degree == 0
                in_scale = True
            else:
                raw_pitch = root_pc + offset
                pad_octave = octave + raw_pitch // 12
                is_root = offset % 12 == 0
                in_scale = offset % 12 in scale_intervals

            rendered_row.append(
                {
                    "note": PC_TO_NOTE[raw_pitch % 12],
                    "octave": pad_octave,
                    "is_root": is_root,
                    "in_scale": in_scale,
                }
            )
        grid.append(rendered_row)
    return grid


async def local_ai_advisor(notes: list[str], emotion: str, theory: dict[str, Any]) -> str:
    prompt = (
        "El músico ha seleccionado estas notas: " + ", ".join(notes) + ".\n"
        f"Análisis determinista: {format_theory(theory)}.\n"
        f"Emoción u objetivo: {emotion!r}.\n"
        "Propón 3 progresiones concretas con cifrado y notas, explica qué escala usar "
        "para melodía/improvisación y sugiere una transición. Sé preciso y responde en español."
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asesor experto en armonía, composición y producción musical. "
                    "No contradigas el análisis de notas proporcionado salvo que expliques el motivo."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.55,
        "max_tokens": 1200,
        "stream": False,
    }
    if AI_MODEL:
        payload["model"] = AI_MODEL
    timeout = aiohttp.ClientTimeout(total=MODEL_REQUEST_TIMEOUT_SECONDS, connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(AI_URL, json=payload, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f"El motor local devolvió HTTP {response.status}")
            data = await response.json()
            return data["choices"][0]["message"]["content"].strip()


def is_loopback_client(address: str | None) -> bool:
    if not address:
        return False
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address == "localhost"


def create_app(
    advisor: Advisor | None = None, lan_token: str | None = None
) -> FastAPI:
    app = FastAPI(title="Cerebro Composer", version="2.0")
    if not lan_token:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    selected_advisor = advisor or local_ai_advisor
    selected_lan_token = lan_token.strip() if lan_token else ""

    @app.middleware("http")
    async def restrict_lan_access(request: Request, call_next):
        client_address = request.client.host if request.client else None
        if is_loopback_client(client_address):
            return await call_next(request)
        if not selected_lan_token:
            return PlainTextResponse("Composer requiere una conexión local.", status_code=403)

        if (request.method, request.url.path) in {("GET", "/connect"), ("POST", "/api/lan-session")}:
            return await call_next(request)

        session_key = request.cookies.get("composer_lan_session", "")
        if not hmac.compare_digest(session_key, selected_lan_token):
            return PlainTextResponse("Composer requiere una clave local.", status_code=403)

        return await call_next(request)

    @app.middleware("http")
    async def add_browser_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; "
            "frame-ancestors 'none'; img-src 'self' data:; object-src 'none'; "
            "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        )
        response.headers.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=(self)")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    def load_ui() -> str:
        with open(UI_PATH, "r", encoding="utf-8") as handle:
            return handle.read()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(load_ui())

    @app.get("/composer", response_class=HTMLResponse)
    async def composer() -> HTMLResponse:
        return HTMLResponse(load_ui())

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon_svg() -> Response:
        return Response(
            content=FAVICON_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def legacy_favicon() -> RedirectResponse:
        return RedirectResponse(url="/favicon.svg", status_code=307)

    @app.get("/assets/composer_audio.js", include_in_schema=False)
    async def composer_audio_detector() -> FileResponse:
        if not os.path.isfile(COMPOSER_AUDIO_PATH):
            raise HTTPException(status_code=404, detail="Detector de audio no disponible")
        return FileResponse(
            COMPOSER_AUDIO_PATH,
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/connect", response_class=HTMLResponse)
    async def lan_connect() -> HTMLResponse:
        return HTMLResponse(LAN_CONNECT_PAGE)

    @app.post("/api/lan-session", status_code=204)
    async def lan_session(request: Request) -> Response:
        supplied_key = request.headers.get("X-Composer-Access-Key", "")
        if not selected_lan_token or not hmac.compare_digest(supplied_key, selected_lan_token):
            raise HTTPException(status_code=403, detail="Clave local no válida")
        response = Response(status_code=204)
        response.set_cookie(
            "composer_lan_session",
            selected_lan_token,
            httponly=True,
            samesite="strict",
            max_age=8 * 60 * 60,
            path="/",
        )
        return response

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        ai_state = "offline"
        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(models_endpoint(AI_URL), allow_redirects=False) as response:
                    if response.status == 200:
                        ai_state = "online"
        except (aiohttp.ClientError, TimeoutError):
            pass
        return HealthResponse(
            status="ok",
            ai=ai_state,
            ui="online" if os.path.isfile(UI_PATH) else "missing",
        )

    @app.post("/api/export/midi")
    async def export_midi(request: MidiExportRequest) -> Response:
        """Devuelve una secuencia MIDI local creada a partir de acordes confirmados."""
        try:
            midi_data = build_midi_file(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=midi_data,
            media_type="audio/midi",
            headers={
                "Content-Disposition": 'attachment; filename="composer-progression.mid"',
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/import/midi")
    async def import_midi(request: Request) -> dict[str, Any]:
        """Importa un archivo MIDI local pequeño sin almacenarlo en el servidor."""
        declared_size = request.headers.get("content-length")
        if declared_size and declared_size.isdigit() and int(declared_size) > MAX_MIDI_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail="El MIDI supera el límite local de 2 MB")
        midi_data = await request.body()
        if not midi_data:
            raise HTTPException(status_code=422, detail="Selecciona un archivo MIDI")
        if len(midi_data) > MAX_MIDI_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail="El MIDI supera el límite local de 2 MB")
        try:
            return parse_midi_file(midi_data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/progression")
    async def progression(request: ProgressionRequest) -> dict[str, Any]:
        """Construye una progresión MIDI completa desde un acorde ya elegido."""
        try:
            return build_progression(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/chord-analysis")
    async def chord_analysis(request: ComposerRequest) -> dict[str, Any]:
        """Analiza y propone cierres sin esperar ni depender del modelo local."""
        intervals: tuple[int, ...] | None = None
        if request.tonic and request.scale:
            intervals = resolve_scale_intervals(request.scale, request.scale_intervals)
        return {
            "theory": analyze_notes(
                request.notes,
                tonic=request.tonic,
                scale_intervals=intervals,
            )
        }

    @app.get("/api/scales")
    async def scales() -> dict[str, Any]:
        return {
            "scales": [
                {
                    "id": identifier,
                    "label": definition["label"],
                    "family": definition["family"],
                    "intervals": list(definition["intervals"]),
                }
                for identifier, definition in SCALE_CATALOG.items()
            ]
        }

    @app.get("/api/push-grid")
    async def push_grid(
        root: str = Query(default="C"),
        scale: str = Query(default="major"),
        intervals: str | None = Query(default=None),
        mode: str = Query(default="in-key"),
        octave: int = Query(default=3, ge=0, le=7),
    ) -> dict[str, Any]:
        canonical_root = root.strip().upper().replace("♯", "#").replace("♭", "B")
        if canonical_root not in NOTE_TO_PC:
            raise HTTPException(status_code=422, detail="Tónica no válida")
        if mode not in PUSH_MODES:
            raise HTTPException(status_code=422, detail="Modo Push no válido")

        try:
            scale_intervals = resolve_scale_intervals(scale, intervals)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        canonical_root = PC_TO_NOTE[NOTE_TO_PC[canonical_root]]
        root_pc = NOTE_TO_PC[canonical_root]
        return {
            "root": canonical_root,
            "scale": scale,
            "mode": mode,
            "intervals": list(scale_intervals),
            "scale_notes": [PC_TO_NOTE[(root_pc + interval) % 12] for interval in scale_intervals],
            "grid": build_push_grid(canonical_root, scale_intervals, mode, octave),
        }

    @app.post("/api/compose")
    async def compose(request: ComposerRequest) -> dict[str, Any]:
        intervals: tuple[int, ...] | None = None
        if request.tonic and request.scale:
            intervals = resolve_scale_intervals(request.scale, request.scale_intervals)
        theory = analyze_notes(request.notes, tonic=request.tonic, scale_intervals=intervals)
        if intervals is not None:
            scale_label = (
                "Personalizada" if request.scale == "custom" else SCALE_CATALOG[request.scale]["label"]
            )
            theory["performance_context"] = {
                "root": request.tonic,
                "scale": request.scale,
                "intervals": list(intervals),
                "label": f"{request.tonic} {scale_label}",
            }
        baseline = format_theory(theory)
        try:
            advice = await selected_advisor(request.notes, request.emotion, theory)
            response = f"{baseline}\n\nConsejo de Cerebro:\n{advice}"
            ai_status = "online"
        except Exception as exc:
            # La teoría básica sigue funcionando aunque el modelo local esté apagado.
            response = (
                f"{baseline}\n\n"
                "El análisis musical básico está disponible, pero la ampliación de IA "
                f"no respondió: {type(exc).__name__}."
            )
            ai_status = "offline"
        return {"response": response, "theory": theory, "ai_status": ai_status}

    return app


LAN_TOKEN = load_lan_token()
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8082,
        log_level="info",
    )
