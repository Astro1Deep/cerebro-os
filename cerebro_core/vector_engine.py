"""Vector engine using LanceDB + bge-m3 embeddings (native MPS, Ollama fallback)."""

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
import pyarrow as pa
import lancedb
import requests
import json
import re
import numpy as np

from cerebro_core.local_endpoint import local_http_session, validate_loopback_endpoint

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

EMBED_DIM = 1024
EMBED_MODEL = os.environ.get("CEREBRO_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
EMBED_IDLE_SECONDS = float(os.environ.get("CEREBRO_EMBED_IDLE_SECONDS", "180"))
OLLAMA_EMBED_URL = os.environ.get("CEREBRO_OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
IDENTITY_KEY = b"cerebro.embedding.identity.v1"


def _ollama_embedding_identity() -> Optional[str]:
    """Identify the installed bge-m3 artifact; unavailable or ambiguous fails closed."""
    try:
        endpoint = validate_loopback_endpoint(OLLAMA_EMBED_URL, "/api/embeddings")
        tags_url = endpoint[:-len('/api/embeddings')] + '/api/tags'
        with local_http_session() as session:
            response = session.get(tags_url, timeout=(2, 3), allow_redirects=False, stream=True)
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(65536):
                raw.extend(chunk)
                if len(raw) > 1048576:
                    return None
        payload = json.loads(raw)
        entries = payload.get('models', []) if isinstance(payload, dict) else []
        matches = [m for m in entries if isinstance(m, dict) and m.get('name') in ('bge-m3', 'bge-m3:latest')]
        if len(matches) != 1:
            return None
        digest = matches[0].get('digest')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', digest):
            return None
        return json.dumps({'provider': 'ollama', 'model': 'bge-m3:latest',
                           'digest': digest.lower(), 'dimensions': EMBED_DIM,
                           'preprocess': 'strip-truncate8000-l2-v1'}, sort_keys=True, separators=(',', ':'))
    except (requests.RequestException, ValueError, TypeError, AttributeError, OSError):
        return None


class _EmbeddingBackend:
    """Singleton thread-safe: carga bge-m3 una vez, lo reutiliza y lo libera tras inactividad.

    Antes se cargaba/descargaba el modelo (2.2 GB) en cada embedding: 10-30 s por llamada
    y picos de memoria unificada repetidos. Ahora: carga perezosa + descarga por timer.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._device = "cpu"
        self._timer: Optional[threading.Timer] = None
        self._last_used = 0.0
        self._native_broken_until = 0.0

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(EMBED_MODEL, device=self._device, local_files_only=True)
        return self._model

    def _schedule_unload(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(EMBED_IDLE_SECONDS, self.unload)
        self._timer.daemon = True
        self._timer.start()

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            if time.time() - self._last_used < EMBED_IDLE_SECONDS - 1:
                self._schedule_unload()
                return
            self._model = None
            gc.collect()
            if self._device == "mps":
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
            logger.info("[VectorEngine] Modelo de embeddings liberado por inactividad.")

    def _encode_ollama(self, texts: list[str]) -> list[Optional[list[float]]]:
        try:
            endpoint = validate_loopback_endpoint(OLLAMA_EMBED_URL, "/api/embeddings")
        except ValueError:
            return [None for _ in texts]

        vectors = []
        with local_http_session() as session:
            for text in texts:
                vector = None
                try:
                    response = session.post(
                        endpoint, json={"model": "bge-m3", "prompt": text},
                        timeout=(2, 30), allow_redirects=False,
                    )
                    if response.status_code == 200:
                        raw = np.asarray(response.json().get("embedding"), dtype=float)
                        if raw.shape == (EMBED_DIM,) and np.isfinite(raw).all():
                            magnitude = np.linalg.norm(raw)
                            if magnitude:
                                vector = (raw / magnitude).tolist()
                except (requests.RequestException, ValueError, TypeError):
                    pass
                vectors.append(vector)
        return vectors

    def encode(self, texts: list[str]) -> list[Optional[list[float]]]:
        texts = [(t or "").strip()[:8000] for t in texts]
        if not texts:
            return []

        # Never switch embedding spaces on failure: the persistent index may
        # already contain bge-m3 vectors, and E5 is not compatible with them.
        ollama_vecs = self._encode_ollama(texts)
        if ollama_vecs and all(v is not None for v in ollama_vecs):
            return [v for v in ollama_vecs]
        logger.warning("[VectorEngine] bge-m3 no disponible; no se usa otro modelo de embeddings.")
        return ollama_vecs


_BACKEND = _EmbeddingBackend()


def _fit_dim(vector: Optional[list[float]]) -> Optional[list[float]]:
    if vector is None or len(vector) != EMBED_DIM:
        return None
    try:
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (EMBED_DIM,) or not np.isfinite(values).all() or not np.any(values):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return values.tolist()


class FaissHotCache:
    """
    Tiered Memory L1 Cache powered by FAISS (In-Memory HNSW / FlatIP).
    Proporciona búsqueda semántica a nivel de microsegundos en RAM unificada M4 Pro,
    mientras que LanceDB garantiza persistencia transaccional L2 sin copias innecesarias.
    """
    def __init__(self, dim: int = EMBED_DIM, max_elements: int = 4000):
        self.dim = dim
        self.max_elements = max_elements
        self.enabled = _FAISS_AVAILABLE
        self._lock = threading.RLock()
        self.id_to_int: dict[str, int] = {}
        self.int_to_record: dict[int, dict[str, Any]] = {}
        self.counter = 1
        self.index = None
        if self.enabled:
            try:
                base_index = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIDMap(base_index)
            except Exception as e:
                logger.warning("[FaissHotCache] No se pudo inicializar índice FAISS: %s", e)
                self.enabled = False

    def size(self) -> int:
        with self._lock:
            return self.index.ntotal if (self.enabled and self.index) else len(self.id_to_int)

    def add(self, doc_id: str, vector: list[float], record: dict[str, Any]) -> None:
        if not self.enabled or not self.index:
            return
        if not vector or len(vector) != self.dim:
            return
        with self._lock:
            if doc_id in self.id_to_int:
                self.remove(doc_id)
            int_id = self.counter
            self.counter += 1
            self.id_to_int[doc_id] = int_id
            self.int_to_record[int_id] = record

            v = np.array([vector], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            ids = np.array([int_id], dtype=np.int64)
            try:
                self.index.add_with_ids(v, ids)
            except Exception as exc:
                logger.debug("[FaissHotCache] Error agregando a FAISS: %s", exc)

    def remove(self, doc_id: str) -> None:
        if not self.enabled or not self.index:
            return
        with self._lock:
            if doc_id in self.id_to_int:
                int_id = self.id_to_int.pop(doc_id)
                self.int_to_record.pop(int_id, None)
                try:
                    self.index.remove_ids(np.array([int_id], dtype=np.int64))
                except Exception:
                    pass

    def search(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if not self.enabled or not self.index or self.index.ntotal == 0:
            return []
        if not vector or len(vector) != self.dim:
            return []
        with self._lock:
            v = np.array([vector], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            k = min(top_k, self.index.ntotal)
            try:
                scores, ids = self.index.search(v, k)
            except Exception as exc:
                logger.debug("[FaissHotCache] Error buscando en FAISS: %s", exc)
                return []

            results = []
            for score, int_id in zip(scores[0], ids[0]):
                if int_id in self.int_to_record:
                    rec = dict(self.int_to_record[int_id])
                    rec["score"] = float(score)
                    results.append(rec)
            return results


class VectorEngine:
    def __init__(self, root: str | Path = "~/.mlx-stack/knowledge/lancedb", default_table: str = "cerebro_unified_memory"):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.default_table = default_table
        self.db = lancedb.connect(str(self.root))
        
        # Tiered Memory Architecture: FAISS L1 Hot-Cache (RAM) + LanceDB L2 Persistent (Disk)
        self.hot_cache = FaissHotCache(dim=1024)
        
        # Ollama bge-m3 uses 1024 dimensions
        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1024)),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("created_at", pa.float64()),
            pa.field("metadata_json", pa.string()),
        ])
        # Legacy tables have no verified embedding identity; never warm them.

    def _warmup_hot_cache(self, table_name: Optional[str] = None):
        if not self.hot_cache.enabled:
            return
        try:
            table = self.get_table(table_name)
            if table.count_rows() == 0:
                return
            arrow_table = table.to_arrow()
            if arrow_table.num_rows == 0:
                return
            max_warmup = min(1000, arrow_table.num_rows)
            for i in range(max_warmup):
                row = arrow_table.slice(i, 1).to_pydict()
                doc_id = str(row["id"][0])
                vec = list(row["vector"][0])
                meta_str = row.get("metadata_json", ["{}"])[0]
                meta = {}
                try:
                    meta = json.loads(meta_str) if meta_str else {}
                except Exception:
                    pass
                record = {
                    "id": doc_id,
                    "text": str(row.get("text", [""])[0]),
                    "source": str(row.get("source", [""])[0]),
                    "created_at": float(row.get("created_at", [0.0])[0]),
                    "metadata": meta
                }
                self.hot_cache.add(doc_id, vec, record)
            logger.info("[VectorEngine] FAISS L1 Hot-Cache precargado con %d registros.", max_warmup)
        except Exception as exc:
            logger.debug("[VectorEngine] Warmup omitido: %s", exc)

    def get_stats(self) -> dict[str, Any]:
        return {
            "faiss_enabled": False,
            "hot_cache_size": 0,
            "lancedb_root": str(self.root),
            "default_table": self.default_table
        }
        
    def get_table(self, table_name: Optional[str] = None, *, identity: Optional[str] = None, create: bool = False):
        if not identity:
            raise ValueError("Embedding identity is required")
        tname = table_name or self.default_table
        if hasattr(self.db, "list_tables"):
            res = self.db.list_tables()
            tables = res.tables if hasattr(res, "tables") else list(res)
        else:
            tables = self.db.table_names()
        if tname in tables:
            table = self.db.open_table(tname)
        elif create:
            schema = self.schema.with_metadata({IDENTITY_KEY: identity.encode('utf-8')})
            table = self.db.create_table(tname, schema=schema, exist_ok=True)
        else:
            raise ValueError("Embedding index is not available")
        if (table.schema.metadata or {}).get(IDENTITY_KEY) != identity.encode('utf-8'):
            raise ValueError("Embedding index identity is missing or mismatched")
        return table

    def _identified_vector(self, text: Optional[str], vector: Optional[list[float]],
                           vector_identity: Optional[str]):
        before = _ollama_embedding_identity()
        if not before:
            return None, None
        if vector is not None:
            # Callers supplying their own vector must declare its exact identity.
            if vector_identity != before:
                return None, None
        elif text:
            vector = self.get_embedding(text)
        else:
            return None, None
        vector = _fit_dim(vector)
        after = _ollama_embedding_identity()
        if vector is None or before != after:
            return None, None
        return vector, before

    def get_embedding(self, text: str) -> Optional[list[float]]:
        if not text or not text.strip():
            return None
        vecs = _BACKEND.encode([text])
        vector = _fit_dim(vecs[0] if vecs else None)
        if vector is None:
            logger.error("[VectorEngine] No se pudo generar embedding (nativo y fallback fallaron).")
        return vector

    def get_embeddings(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embedding por lotes: una sola pasada de GPU para N textos."""
        clean = [t for t in texts if t and t.strip()]
        if not clean:
            return []
        return [_fit_dim(v) for v in _BACKEND.encode(clean)]

    def upsert(self, doc_id: str, vector: Optional[list[float]] = None, text: Optional[str] = None, table_name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None, *, vector_identity: Optional[str] = None) -> bool:
        if not doc_id:
            return False
            
        vector, identity = self._identified_vector(text, vector, vector_identity)
        if vector is None or identity is None:
            return False
        try:
            table = self.get_table(table_name, identity=identity, create=True)
        except (ValueError, OSError) as exc:
            logger.warning("[VectorEngine] Index blocked: %s", exc)
            return False
        record = {"id": str(doc_id), "vector": vector, "text": text or "", "source": "explicit_user", "created_at": 0.0, "metadata_json": "{}"}
        
        if metadata:
            if "source" in metadata: record["source"] = str(metadata["source"])
            if "created_at" in metadata: record["created_at"] = float(metadata["created_at"])
            
            # Extract additional metadata for JSON
            json_meta = {}
            for k, v in metadata.items():
                if k not in ["source", "created_at"]:
                    json_meta[k] = v
            if json_meta:
                record["metadata_json"] = json.dumps(json_meta, ensure_ascii=False)

        try:
            # A single merge avoids deleting the old record before its replacement
            # has been committed. Unsupported backends fail without that deletion.
            (table.merge_insert("id").when_matched_update_all()
             .when_not_matched_insert_all().execute([record]))
        except Exception as exc:
            logger.error("[VectorEngine] Failed to add record %s: %s", doc_id, exc)
            return False
        return True

    def search(self, vector: Optional[list[float]] = None, text: Optional[str] = None, limit: int = 5, table_name: Optional[str] = None, *, vector_identity: Optional[str] = None) -> list[dict[str, Any]]:
        vector, identity = self._identified_vector(text, vector, vector_identity)
        if vector is None or identity is None:
            return []

        # 1. Búsqueda L1 Hot-Cache FAISS (~0.05ms)
        hot_results = []
            
        # 2. Búsqueda L2 persistente LanceDB
        try:
            table = self.get_table(table_name, identity=identity)
        except (ValueError, OSError) as exc:
            logger.warning("[VectorEngine] Index blocked: %s", exc)
            return []
        lancedb_results = []
        try:
            if table.count_rows() > 0:
                lancedb_results = table.search(vector, vector_column_name="vector").metric("cosine").limit(limit).to_list()
        except Exception as exc:
            logger.error("[VectorEngine] LanceDB search error: %s", exc)

        # 3. Fusión Híbrida con deduplicación por ID
        merged_by_id: dict[str, dict[str, Any]] = {}
        for r in lancedb_results:
            dist = float(r.get("_distance", 0.0))
            doc_id = str(r.get("id", ""))
            item = {
                "id": doc_id,
                "score": float(1.0 - dist),
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "created_at": r.get("created_at", 0.0)
            }
            try:
                meta_str = r.get("metadata_json", "{}")
                item["metadata"] = json.loads(meta_str) if meta_str else {}
            except Exception:
                item["metadata"] = {}
            merged_by_id[doc_id] = item

        for hr in hot_results:
            doc_id = hr["id"]
            if doc_id not in merged_by_id or hr["score"] > merged_by_id[doc_id]["score"]:
                merged_by_id[doc_id] = hr

        sorted_results = sorted(merged_by_id.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]

    def delete(self, doc_id: str, table_name: Optional[str] = None) -> bool:
        if not doc_id: return False
        clean_id = str(doc_id).replace("'", "''")
        identity = _ollama_embedding_identity()
        if not identity:
            return False
        try:
            table = self.get_table(table_name, identity=identity)
            table.delete(f"id = '{clean_id}'")
            if table_name is None or table_name == self.default_table:
                self.hot_cache.remove(str(doc_id))
            return True
        except Exception:
            return False
