"""
RAG ingestion — turns uploaded documents into chunks + embeddings and
stores them in the Knowledge Base (backend/memory/vectordb.py).

This turns document text into chunks, embeds each chunk, and persists the
results in the local vector store. It is intentionally small and fast so it
works well for uploads without needing an external database service.
"""

from backend.config import settings
from backend.logging_config import get_logger
from backend.memory.vectordb import vector_db
from backend.rag.text import split_text

log = get_logger(__name__)


def ingest_document(text: str, source: str) -> dict[str, object]:
    chunks = split_text(
        text,
        max_chars=settings.RAG_CHUNK_SIZE,
        overlap=settings.RAG_CHUNK_OVERLAP,
    )
    if not chunks:
        return {
            "source": source,
            "chunks_added": 0,
            "chunk_ids": [],
        }

    metadatas = [
        {
            "source": source,
            "chunk_index": index,
            "chunk_count": len(chunks),
        }
        for index in range(len(chunks))
    ]
    chunk_ids = vector_db.add(chunks, metadatas)
    log.info("[rag] ingested source=%s chunks=%d", source, len(chunk_ids))
    return {
        "source": source,
        "chunks_added": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }
