"""Evidence-bound commercial proposals awaiting explicit owner approval."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from uuid import uuid4

from cerebro_core.web_knowledge import WebKnowledgeError, WebKnowledgeStore


class EvidenceProposalBoard:
    """Keeps commercial hypotheses separate from real approved source evidence."""

    def __init__(self, database_path: str | Path, knowledge_store: WebKnowledgeStore) -> None:
        self.database_path = Path(database_path)
        self.knowledge_store = knowledge_store
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.database_path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS commercial_proposals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending_approval', 'approved', 'rejected')),
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    approver TEXT
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _approved_sources(self, source_ids: list[str]) -> list[dict[str, object]]:
        unique_ids = list(dict.fromkeys(str(item).strip() for item in source_ids if str(item).strip()))
        if len(unique_ids) < 2:
            raise WebKnowledgeError("una propuesta comercial requiere al menos dos fuentes aprobadas")
        sources: list[dict[str, object]] = []
        for identifier in unique_ids:
            document = self.knowledge_store.get_document(identifier)
            if document["status"] != "approved":
                raise WebKnowledgeError("todas las fuentes de una propuesta deben estar aprobadas")
            sources.append(
                {
                    "id": document["id"],
                    "url": document["url"],
                    "title": document["title"],
                    "sha256": document["sha256"],
                    "fetched_at": document["fetched_at"],
                    "approved_at": document["approved_at"],
                }
            )
        if len({str(source["url"]) for source in sources}) < 2:
            raise WebKnowledgeError("una propuesta comercial requiere dos fuentes distintas")
        return sources

    def create_pending(self, *, title: str, hypothesis: str, source_ids: list[str]) -> dict[str, object]:
        clean_title = title.strip()
        clean_hypothesis = hypothesis.strip()
        if not clean_title or not clean_hypothesis:
            raise ValueError("la propuesta requiere título e hipótesis")
        if len(clean_title) > 300 or len(clean_hypothesis) > 10_000:
            raise ValueError("la propuesta supera el límite de tamaño")
        sources = self._approved_sources(source_ids)
        identifier = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO commercial_proposals (
                    id, title, hypothesis, sources_json, status, created_at, approved_at, approver
                ) VALUES (?, ?, ?, ?, 'pending_approval', ?, NULL, NULL)
                """,
                (identifier, clean_title, clean_hypothesis, json.dumps(sources, ensure_ascii=False), self._now()),
            )
        return self.get(identifier)

    def get(self, identifier: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, hypothesis, sources_json, status, created_at, approved_at, approver
                FROM commercial_proposals WHERE id = ?
                """,
                (identifier,),
            ).fetchone()
        if row is None:
            raise ValueError("no existe esa propuesta comercial")
        proposal = dict(row)
        proposal["sources"] = json.loads(str(proposal.pop("sources_json")))
        return proposal

    def list_recent(self, *, limit: int = 10) -> list[dict[str, object]]:
        """Return proposal metadata without exposing unbounded source contents."""

        if not 1 <= limit <= 100:
            raise ValueError("límite de propuestas no válido")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, status, created_at, approved_at, approver
                FROM commercial_proposals
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def approve(self, identifier: str, *, approver: str) -> dict[str, object]:
        clean_approver = approver.strip()
        if not clean_approver:
            raise ValueError("debe identificarse quién aprueba")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE commercial_proposals
                SET status = 'approved', approved_at = ?, approver = ?
                WHERE id = ? AND status = 'pending_approval'
                """,
                (self._now(), clean_approver[:128], identifier),
            )
        if cursor.rowcount == 0:
            current = self.get(identifier)
            if current["status"] == "approved":
                return current
            raise ValueError("la propuesta no está pendiente de aprobación")
        return self.get(identifier)
