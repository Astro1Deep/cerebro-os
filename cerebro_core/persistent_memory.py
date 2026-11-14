from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any

from cerebro_core.vector_engine import VectorEngine

logger = logging.getLogger(__name__)

class PersistentMemoryError(ValueError):
    """An explicit local memory entry could not be processed."""

def _get_embedding_model():
    """Compatibility helper for legacy scripts and migration tests."""
    class EmbeddingModelWrapper:
        def encode(self, texts, batch_size=None, **kwargs):
            import numpy as np
            ve = VectorEngine()
            vectors = []
            if isinstance(texts, str):
                texts = [texts]
            for t in texts:
                vec = ve.get_embedding(t) or [0.0] * 1024
                vectors.append(vec)
            return np.array(vectors)
    return EmbeddingModelWrapper()

class PersistentMemoryStore:
    """
    Almacena hechos de usuario y permite búsqueda RAG semántica de conversaciones históricas.
    Integra LanceDB para la búsqueda vectorial (Ollama) y SQLite para retención rápida.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except Exception:
            pass

        self.database_path = self.root / "user_memory.db"
        self._initialise()

        # LanceDB setup
        self.vector_engine = VectorEngine(root=self.root / "lancedb") # Utiliza la tabla cerebro_unified_memory por defecto

        self._last_timestamp: float = 0.0
        self._lock = threading.Lock()
        self._enforce_permissions()

    def _connect(self) -> sqlite3.Connection:
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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        id UNINDEXED,
                        text,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
                row_fts = connection.execute("SELECT count(*) FROM memories_fts").fetchone()
                row_mem = connection.execute("SELECT count(*) FROM memories").fetchone()
                if row_fts and row_mem and row_fts[0] < row_mem[0]:
                    connection.execute("INSERT OR IGNORE INTO memories_fts(id, text) SELECT id, text FROM memories")
            except Exception as e:
                logger.debug("FTS5 table initialization: %s", e)
        if self.database_path.exists():
            try:
                os.chmod(self.database_path, 0o600)
            except Exception:
                pass

    def _enforce_permissions(self) -> None:
        try:
            if self.root.exists():
                os.chmod(self.root, 0o700)
            if self.database_path.exists():
                os.chmod(self.database_path, 0o600)
            for db_extra in (self.root / "user_memory.db-wal", self.root / "user_memory.db-shm"):
                if db_extra.exists():
                    os.chmod(db_extra, 0o600)
        except Exception:
            pass

    def remember(self, text: str, *, source: str = "explicit_user") -> dict[str, object]:
        if not isinstance(text, str):
            raise PersistentMemoryError("la memoria debe contener entre 3 y 2.000 caracteres")
        clean = " ".join(text.split())
        if not 3 <= len(clean) <= 2_000:
            raise PersistentMemoryError("la memoria debe contener entre 3 y 2.000 caracteres")

        # Validación estricta de origen
        if not isinstance(source, str) or source in ("system", "bot", "assistant", "automated", "") or source.strip() == "":
            raise PersistentMemoryError(f"Origen no permitido: '{source}'")

        with self._lock:
            timestamp = datetime.now(timezone.utc).timestamp()
            if timestamp <= self._last_timestamp:
                timestamp = self._last_timestamp + 0.000001
            self._last_timestamp = timestamp

        identifier = hashlib.sha256(f"{clean}\n{timestamp:.6f}".encode("utf-8")).hexdigest()[:16]

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memories(id, text, source, created_at) VALUES (?, ?, ?, ?)",
                (identifier, clean, source, timestamp),
            )
            try:
                connection.execute(
                    "INSERT OR REPLACE INTO memories_fts(id, text) VALUES (?, ?)",
                    (identifier, clean),
                )
            except Exception:
                pass

        # Guarda en LanceDB
        success = self.vector_engine.upsert(
            doc_id=identifier,
            text=clean,
            metadata={"source": source, "created_at": float(timestamp)}
        )
        if not success:
            logger.error("Error guardando vector en LanceDB para '%s'", identifier)

        self._enforce_permissions()
        return {"id": identifier, "text": clean, "source": source, "created_at": timestamp}

    def _bm25_search(self, query: str, *, limit: int = 15) -> list[dict[str, Any]]:
        clean_tokens = [w for w in re.findall(r"[\wÀ-ÿ0-9_-]{2,}", query) if w.strip()]
        if not clean_tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in clean_tokens)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT m.id, m.text, m.source, m.created_at, bm25(memories_fts) AS bm25_rank
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.id
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25_rank ASC
                    LIMIT ?
                    """,
                    (fts_query, limit)
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.debug("[PersistentMemoryStore] BM25 FTS5 search error: %s", exc)
            return []

    def search_rrf(self, query: str, *, limit: int = 4, rrf_k: int = 60) -> list[dict[str, object]]:
        """
        Búsqueda Híbrida RRF (Reciprocal Rank Fusion).
        Fusiona el ranking denso de vectores LanceDB/FAISS con el ranking léxico BM25 de SQLite FTS5.
        Fórmula: RRF_score(d) = sum( 1 / (k + rank_m(d)) ) con k=60.
        Garantiza máxima precisión tanto en código fuente y símbolos exactos como en canon conceptual.
        """
        if not isinstance(query, str):
            return []
        query_clean = query.strip()
        if not query_clean or limit <= 0:
            return []

        candidate_k = max(limit * 3, 15)
        # 1. Búsqueda vectorial densa (FAISS L1 Hot Cache + LanceDB L2)
        vector_results = self.vector_engine.search(text=query_clean, limit=candidate_k)

        # 2. Búsqueda léxica dispersa BM25 (SQLite FTS5)
        bm25_results = self._bm25_search(query_clean, limit=candidate_k)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        items_by_id: dict[str, dict[str, Any]] = {}

        for rank, vr in enumerate(vector_results, start=1):
            doc_id = str(vr.get("id"))
            items_by_id[doc_id] = {
                "id": doc_id,
                "text": vr.get("text", ""),
                "source": vr.get("source", "memoria_historica"),
                "created_at": float(vr.get("created_at", 0.0)),
                "vector_score": float(vr.get("score", 0.0)),
                "bm25_score": None,
            }
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, br in enumerate(bm25_results, start=1):
            doc_id = str(br.get("id"))
            if doc_id not in items_by_id:
                items_by_id[doc_id] = {
                    "id": doc_id,
                    "text": br.get("text", ""),
                    "source": br.get("source", "memoria_historica"),
                    "created_at": float(br.get("created_at", 0.0)),
                    "vector_score": 0.0,
                    "bm25_score": float(br.get("bm25_rank", 0.0)),
                }
            else:
                items_by_id[doc_id]["bm25_score"] = float(br.get("bm25_rank", 0.0))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        if not rrf_scores:
            return []

        sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        final_memories = []
        for did in sorted_ids[:limit]:
            entry = items_by_id[did]
            final_memories.append({
                "id": entry["id"],
                "text": entry["text"],
                "source": entry["source"],
                "score": float(round(rrf_scores[did], 6)),
                "rrf_score": float(round(rrf_scores[did], 6)),
                "vector_score": entry["vector_score"],
                "bm25_score": entry["bm25_score"],
                "created_at": entry["created_at"],
            })

        return final_memories

    def search_hybrid(self, query: str, *, limit: int = 4, rrf_k: int = 60) -> list[dict[str, object]]:
        """Alias para search_rrf."""
        return self.search_rrf(query, limit=limit, rrf_k=rrf_k)

    def search(self, query: str, *, limit: int = 4, mode: str = "vector", rrf_k: int = 60) -> list[dict[str, object]]:
        if mode in ("rrf", "hybrid"):
            return self.search_rrf(query, limit=limit, rrf_k=rrf_k)

        if not isinstance(query, str):
            return []
        query_clean = query.strip()
        if not query_clean or limit <= 0:
            return []

        results = self.vector_engine.search(text=query_clean, limit=limit)
        
        memories = []
        for r in results:
            source_val = r.get("source", "memoria_historica")
            created_val = float(r.get("created_at", 0.0))
            item = {
                "id": r.get("id"),
                "text": r.get("text"),
                "source": source_val,
                "score": r.get("score", 0.0),
                "created_at": created_val
            }
            memories.append(item)

        return memories

    def list_recent(self, *, limit: int = 20) -> list[dict[str, object]]:
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise PersistentMemoryError("El límite de recuerdos recientes debe ser entre 1 y 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, source, created_at FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, identifier: str) -> bool:
        if not identifier or not isinstance(identifier, str):
            return False

        found = False
        with self._connect() as connection:
            cur = connection.execute("DELETE FROM memories WHERE id = ?", (identifier,))
            if cur.rowcount > 0:
                found = True
                try:
                    connection.execute("DELETE FROM memories_fts WHERE id = ?", (identifier,))
                except Exception:
                    pass

        if found:
            self.vector_engine.delete(doc_id=identifier)

        self._enforce_permissions()
        return found
