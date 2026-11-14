"""Read local BeatOS bridge metadata and query advertised audio indexes."""

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

SCHEMA = "cerebro-beatos-bridge-v2"


def read_beatos_bridge() -> dict:
    configured = os.environ.get("CEREBRO_BEATOS_BRIDGE")
    path = Path(configured).expanduser() if configured else Path.home() / ".mlx-stack" / "beatos_bridge.json"
    try:
        with path.open("rb") as stream:
            payload = json.loads(stream.read(1024 * 1024 + 1))
    except (OSError, ValueError, UnicodeError):
        return {}
    return payload if isinstance(payload, dict) and payload.get("schema") == SCHEMA else {}


def beatos_api_base() -> str:
    value = read_beatos_bridge().get("api_base")
    if isinstance(value, str):
        try:
            url = urlsplit(value)
            if url.scheme == "http" and url.hostname == "127.0.0.1" and url.port and not (
                url.username or url.password or url.query or url.fragment or url.path not in ("", "/")
            ):
                return f"http://127.0.0.1:{url.port}"
        except ValueError:
            pass
    return "http://127.0.0.1:8888"


def get_dynamic_disks_config() -> dict[str, dict[str, str]]:
    bridge = read_beatos_bridge()
    entries = bridge.get("mounted_libraries")
    libraries = [{
        "id": "mpc",
        "name": "BeatOS",
        "db_path": bridge.get("database_path"),
        "mount_path": bridge.get("root_dir"),
        "table_name": "samples",
    }]
    if isinstance(entries, list):
        libraries.extend(entries[:100])
    result = {}
    paths = set()
    for library in libraries:
        if not isinstance(library, dict):
            continue
        key, database = library.get("id"), library.get("db_path")
        if not isinstance(key, str) or not key or key in result or not isinstance(database, str):
            continue
        path = Path(database).expanduser()
        try:
            if not path.is_absolute() or not path.is_file():
                continue
            path = path.resolve()
        except (OSError, ValueError):
            continue
        if path in paths:
            continue
        paths.add(path)
        name, mount, table = (library.get(field) for field in ("name", "mount_path", "table_name"))
        result[key] = {
            "name": name if isinstance(name, str) else key,
            "mount": mount if isinstance(mount, str) else "",
            "db": str(path),
            "table": table if isinstance(table, str) and table else "",
        }
    return result


def library_status() -> dict:
    items = []
    for key, config in get_dynamic_disks_config().items():
        mount = Path(config["mount"]) if config["mount"] else Path(config["db"]).parent
        record = {"key": key, "label": config["name"], "name": config["name"],
                  "role": "BeatOS", "mount": str(mount), "index_available": True,
                  "mounted": False, "total_gb": 0.0, "free_gb": 0.0, "used_pct": 0.0}
        try:
            usage = shutil.disk_usage(mount)
        except (OSError, ValueError):
            pass
        else:
            record.update(mounted=True, total_gb=round(usage.total / 1024**3, 1),
                          free_gb=round(usage.free / 1024**3, 1),
                          used_pct=round(usage.used / usage.total * 100, 1) if usage.total else 0.0)
        items.append(record)
    count = sum(bool(item["mounted"]) for item in items)
    return {"mounted_count": count, "total_disks": len(items), "items": items,
            "summary": f"BeatOS: {count}/{len(items)}"}


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def search_samples(
    query: str = "",
    sound_type: Optional[str] = None,
    bpm_min: Optional[int] = None,
    bpm_max: Optional[int] = None,
    key: Optional[str] = None,
    camelot: Optional[str] = None,
    disk: str = "all",
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Busca samples en las bases de datos de los discos externos con filtros musicales.
    """
    results = []
    active_disks = get_dynamic_disks_config()
    limit = max(0, min(int(limit), 500))
    if not limit:
        return results
    target_disks = list(active_disks) if disk == "all" else [disk] if disk in active_disks else []

    for d_key in target_disks:
        cfg = active_disks[d_key]
        db_path = cfg["db"]
        if not os.path.exists(db_path):
            continue

        try:
            with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
                conn.row_factory = sqlite3.Row
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )}
                table = cfg["table"]
                if table not in tables:
                    table = "samples" if "samples" in tables else None
                if not table:
                    continue
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({quote(table)})")}
                aliases = {
                    "id": ("id",), "filename": ("filename", "name", "title", "track_name", "bank_name"),
                    "full_path": ("full_path", "file_path", "path", "location", "absolute_path", "root_path"),
                    "sound_type": ("sound_type", "category", "asset_type", "sample_type", "genre", "macro_genre"),
                    "sub_pack": ("sub_pack", "pack", "collection", "style_tag"),
                    "bpm": ("bpm", "tempo"), "musical_key": ("musical_key", "key", "initial_key"),
                    "camelot": ("camelot",), "energy_level": ("energy_level",), "size_mb": ("size_mb",),
                }
                selected = {key: next((quote(name) for name in names if name in columns), None)
                            for key, names in aliases.items()}
                if not selected["filename"] or not selected["full_path"]:
                    continue
                fields = ", ".join(f"{value or 'NULL'} AS {quote(key)}" for key, value in selected.items())
                predicates, params = [], []
                if query:
                    parts = [selected[key] for key in ("filename", "sub_pack") if selected[key]]
                    predicates.append("(" + " OR ".join(f"{column} LIKE ?" for column in parts) + ")")
                    params.extend([f"%{query}%"] * len(parts))
                filters = (("sound_type", "COLLATE NOCASE =", sound_type),
                           ("bpm", ">=", bpm_min), ("bpm", "<=", bpm_max),
                           ("musical_key", "LIKE", f"%{key}%" if key else None),
                           ("camelot", "LIKE", f"%{camelot}%" if camelot else None))
                for name, operator, value in filters:
                    if value is not None:
                        predicates.append(f"{selected[name] or 'NULL'} {operator} ?")
                        params.append(value)
                where = " WHERE " + " AND ".join(predicates) if predicates else ""
                rows = conn.execute(f"SELECT {fields} FROM {quote(table)}{where} LIMIT ?", (*params, limit - len(results)))
                results.extend({**dict(row), "disk": d_key, "disk_name": cfg["name"]} for row in rows)
                if len(results) >= limit:
                    break
        except (OSError, sqlite3.Error) as e:
            print(f"Error consultando DB de {d_key}: {e}", file=sys.stderr)

    return results
