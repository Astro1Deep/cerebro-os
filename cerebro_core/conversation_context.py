"""Assemble the bounded local context used for one Cerebro chat request."""

from __future__ import annotations

from pathlib import Path

from cerebro_core.persistent_memory import PersistentMemoryStore
from cerebro_core.system_context import build_system_context
from cerebro_core.web_knowledge import default_store



def lexical_rerank(query, chunks, top_k=3):
    # Pure Python lexical Reranking (Jaccard Similarity)
    # Evita abrumar la ventana de contexto filtrando falsos positivos del Vector Engine
    query_words = set(query.lower().split())
    if not query_words:
        return chunks[:top_k]
        
    scored_chunks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_words = set(text.lower().split())
        intersection = query_words.intersection(chunk_words)
        union = query_words.union(chunk_words)
        score = len(intersection) / len(union) if len(union) > 0 else 0
        scored_chunks.append((score, chunk))
    
    # Sort by score descending and take top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_k]]

def build_chat_context(base_dir: Path | str, query: str) -> str:

    root = Path(base_dir).expanduser().resolve()
    sections = [build_system_context(root)]

    memories = PersistentMemoryStore(root / "knowledge").search(query)
    if memories:
        # FASE 3: Compresión Cognitiva (Re-Ranking Léxico + Vectorial)
        # 1. Filtramos los mejores resultados matemáticamente en lugar de enviarlos todos a lo bruto
        memories = lexical_rerank(query, memories, top_k=5)
        
        mem_str = "[SISTEMA - BÚSQUEDA EN MEMORIA HISTÓRICA RAG (RERANKED) - Memoria explícita del usuario]:\n"
        for i, item in enumerate(memories):
            source = item.get("source", "Desconocido")
            text = item.get("text", "").strip()
            mem_str += f"--- FRAGMENTO {i+1} (Origen: {source}) ---\n{text}\n\n"
        sections.append(mem_str)
    source_context = default_store(root).rag_context(query)
    if source_context:
        sections.append(source_context)
    return "\n\n".join(sections)[:8_000]
