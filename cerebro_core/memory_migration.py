"""Build a separate, resumable vector candidate from a read-only SQLite source.

This never activates a candidate or deletes/relabels an old index. The source is
read in one SQLite transaction, so WAL contents are included in the snapshot.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from cerebro_core.vector_engine import VectorEngine, _ollama_embedding_identity


class MigrationError(RuntimeError):
    pass


def _source_records(source: Path, kind: str) -> list[dict[str, Any]]:
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise MigrationError('Source must be an existing SQLite file')
    uri = source.as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        if kind == 'memory':
            rows = connection.execute('SELECT id, text, source, created_at FROM memories ORDER BY id').fetchall()
            records = [{'id': row[0], 'text': row[1], 'metadata': {'source': row[2], 'created_at': row[3]}} for row in rows]
        elif kind == 'web-approved':
            rows = connection.execute("SELECT id, text, url, title, approved_at FROM documents WHERE status='approved' ORDER BY id").fetchall()
            records = [{'id': row[0], 'text': row[1][:2000], 'metadata': {'source': 'web_knowledge', 'url': row[2], 'title': row[3], 'created_at': row[4] or 0.0}} for row in rows]
        else:
            raise MigrationError('Unknown source kind')
    if any(not isinstance(r['id'], str) or not r['id'] or not isinstance(r['text'], str) or not r['text'].strip() for r in records):
        raise MigrationError('A source record has an invalid id or text')
    if len({r['id'] for r in records}) != len(records):
        raise MigrationError('Source record IDs are not unique')
    return records


def _snapshot_digest(records: list[dict[str, Any]], kind: str) -> str:
    value = json.dumps({'kind': kind, 'records': records}, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def migrate(source: Path, destination: Path, kind: str, *, apply: bool = False) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if destination == source or destination in source.parents or source.parent in (destination, *destination.parents):
        raise MigrationError('Destination must be separate from the source directory')
    records = _source_records(source, kind)
    digest = _snapshot_digest(records, kind)
    report = {'kind': kind, 'source_records': len(records), 'source_snapshot_sha256': digest,
              'applied': False, 'activated': False}
    if not apply:
        return report
    identity = _ollama_embedding_identity()
    if not identity:
        raise MigrationError('Verified local bge-m3 identity is unavailable')
    if destination.exists() and (not destination.is_dir() or (any(destination.iterdir()) and not (destination / 'migration.sqlite3').is_file())):
        raise MigrationError('Destination must be empty or an existing migration candidate')
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_file = destination / 'migration.sqlite3'
    if state_file.is_symlink() or (destination / 'lancedb').is_symlink():
        raise MigrationError('Candidate state and index paths must not be symbolic links')
    with closing(sqlite3.connect(str(state_file), timeout=1)) as state:
        state.execute('CREATE TABLE IF NOT EXISTS migration (singleton INTEGER PRIMARY KEY CHECK(singleton=1), source_digest TEXT NOT NULL, kind TEXT NOT NULL, identity TEXT NOT NULL)')
        state.execute('CREATE TABLE IF NOT EXISTS completed (id TEXT PRIMARY KEY)')
        state.commit()
        try:
            state.execute('BEGIN IMMEDIATE')
            existing = state.execute('SELECT source_digest, kind, identity FROM migration WHERE singleton=1').fetchone()
            expected = (digest, kind, identity)
            if existing and existing != expected:
                raise MigrationError('Source snapshot, kind or model identity changed; start a new candidate')
            if not existing:
                state.execute('INSERT INTO migration VALUES (1,?,?,?)', expected)
            # Persist the immutable contract before any vectors are written, so
            # a crash cannot leave a candidate without a source/model binding.
            state.commit()
            state.execute('BEGIN IMMEDIATE')
            if state.execute('SELECT source_digest, kind, identity FROM migration WHERE singleton=1').fetchone() != expected:
                raise MigrationError('Candidate contract changed')
            engine = VectorEngine(root=destination / 'lancedb', default_table='migration_candidate')
            completed = {row[0] for row in state.execute('SELECT id FROM completed')}
            for item in records:
                if item['id'] in completed:
                    continue
                if _ollama_embedding_identity() != identity:
                    raise MigrationError('Model identity changed during migration')
                if not engine.upsert(doc_id=item['id'], text=item['text'], metadata=item['metadata']):
                    raise MigrationError('Embedding or candidate write failed; resume with the same snapshot and identity')
                state.execute('INSERT OR IGNORE INTO completed(id) VALUES (?)', (item['id'],))
            if _ollama_embedding_identity() != identity:
                raise MigrationError('Model identity changed during migration')
            table = engine.get_table(identity=identity, create=bool(records)) if records else None
            if table is not None and table.count_rows() != len(records):
                raise MigrationError('Candidate count does not match the source snapshot')
            state.commit()
        except Exception:
            state.rollback()
            raise
    report.update(applied=True, candidate_records=len(records), model_identity=identity)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--kind', choices=('memory', 'web-approved'), required=True)
    parser.add_argument('--apply', action='store_true', help='Build candidate only; never activate it')
    args = parser.parse_args()
    try:
        print(json.dumps(migrate(args.source, args.destination, args.kind, apply=args.apply), ensure_ascii=False, indent=2))
        return 0
    except (MigrationError, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({'error': str(exc), 'activated': False}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
