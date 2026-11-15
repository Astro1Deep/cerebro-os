"""Private, non-executing import of supported Flipper Zero text exports.

This is an evidence/import workflow, not a radio, NFC, credential, or cracking
tool.  It accepts a small allowlist of Flipper text formats, stores the original
privately under a generated identifier, and reports only non-sensitive metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Final
from uuid import uuid4


MAX_FLIPPER_FILE_BYTES: Final[int] = 100_000_000
MAX_HEADER_LINES: Final[int] = 128
ALLOWED_SUFFIXES: Final[dict[str, tuple[str, ...]]] = {
    ".nfc": ("flipper nfc",),
    ".sub": ("flipper subghz",),
    ".ir": ("flipper ir",),
    ".rfid": ("flipper rfid",),
    ".ibtn": ("flipper ibutton",),
}
DISPLAYED_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {"filetype", "version", "device type", "frequency", "protocol", "preset", "name", "type"}
)
SENSITIVE_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {"key", "data", "uid", "id", "password", "secret", "token", "private key"}
)


class FlipperImportError(ValueError):
    """The submitted Telegram document is not a supported safe import."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise FlipperImportError("el nombre del archivo no es válido")
    candidate = Path(value)
    if candidate.name != value or any(not character.isprintable() for character in value):
        raise FlipperImportError("el nombre del archivo no es válido")
    suffix = candidate.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise FlipperImportError("formato no permitido; se admiten .nfc, .sub, .ir, .rfid e .ibtn")
    return value


def _parse_headers(payload: bytes, suffix: str) -> tuple[str, dict[str, str], bool]:
    if not payload or len(payload) > MAX_FLIPPER_FILE_BYTES:
        raise FlipperImportError("el archivo debe contener entre 1 y 1.000.000 bytes")
    if b"\x00" in payload:
        raise FlipperImportError("el archivo no es un export de texto Flipper válido")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FlipperImportError("el archivo no es texto UTF-8 de Flipper") from error
    headers: dict[str, str] = {}
    sensitive_present = False
    for line in text.splitlines()[:MAX_HEADER_LINES]:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = " ".join(key.strip().lower().split())
        normalized_value = " ".join(value.strip().split())[:256]
        if normalized_key in SENSITIVE_METADATA_KEYS:
            sensitive_present = True
        elif normalized_key in DISPLAYED_METADATA_KEYS and normalized_value:
            headers[normalized_key] = normalized_value
    filetype = headers.get("filetype", "")
    if not filetype or not any(filetype.lower().startswith(prefix) for prefix in ALLOWED_SUFFIXES[suffix]):
        raise FlipperImportError("el contenido no coincide con el formato Flipper declarado")
    if "version" not in headers:
        raise FlipperImportError("el export de Flipper no declara versión")
    return filetype, headers, sensitive_present


class FlipperImportStore:
    """Stores original owner-submitted files privately and exposes redacted metadata."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.files_dir = self.root / "imports"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.files_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.database_path = self.root / "flipper_imports.sqlite"
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        # MEJORA ARQUITECTÓNICA DE BAJO NIVEL: Concurrencia SQLite (Write-Ahead Logging)
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS imports (
                    id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    filetype TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    sensitive_material_present INTEGER NOT NULL,
                    imported_at TEXT NOT NULL
                )
                """
            )
        os.chmod(self.database_path, 0o600)

    def ingest(self, source_name: object, payload: bytes) -> dict[str, object]:
        name = _safe_name(source_name)
        suffix = Path(name).suffix.lower()
        filetype, metadata, sensitive_present = _parse_headers(payload, suffix)
        identifier = uuid4().hex
        destination = self.files_dir / f"{identifier}{suffix}"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.files_dir, prefix=".upload-", delete=False) as temporary_file:
                temporary_name = temporary_file.name
                os.fchmod(temporary_file.fileno(), 0o600)
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, destination)
            os.chmod(destination, 0o600)
            record = {
                "id": identifier,
                "source_name": name,
                "suffix": suffix,
                "filetype": filetype,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "metadata": metadata,
                "sensitive_material_present": sensitive_present,
                "imported_at": _now(),
            }
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO imports(
                        id, source_name, suffix, filetype, sha256, size_bytes,
                        metadata_json, sensitive_material_present, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        record["source_name"],
                        record["suffix"],
                        record["filetype"],
                        record["sha256"],
                        record["size_bytes"],
                        json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True),
                        int(record["sensitive_material_present"]),
                        record["imported_at"],
                    ),
                )
            return record
        except (OSError, sqlite3.Error) as error:
            destination.unlink(missing_ok=True)
            raise FlipperImportError("el archivo no se pudo almacenar de forma privada") from error
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def list_recent(self, *, limit: int = 10) -> list[dict[str, object]]:
        if not 1 <= limit <= 100:
            raise FlipperImportError("límite de importaciones no válido")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source_name, suffix, filetype, sha256, size_bytes,
                       metadata_json, sensitive_material_present, imported_at
                FROM imports ORDER BY imported_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records: list[dict[str, object]] = []
        for row in rows:
            record = dict(row)
            record["metadata"] = json.loads(str(record.pop("metadata_json")))
            record["sensitive_material_present"] = bool(record["sensitive_material_present"])
            records.append(record)
        return records

    def get(self, identifier: object) -> dict[str, object]:
        """Return one redacted metadata record by its opaque local identifier."""

        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise FlipperImportError("el identificador de importación no es válido")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, source_name, suffix, filetype, sha256, size_bytes,
                       metadata_json, sensitive_material_present, imported_at
                FROM imports WHERE id = ?
                """,
                (identifier,),
            ).fetchone()
        if row is None:
            raise FlipperImportError("no existe esa importación Flipper")
        record = dict(row)
        record["metadata"] = json.loads(str(record.pop("metadata_json")))
        record["sensitive_material_present"] = bool(record["sensitive_material_present"])
        return record
