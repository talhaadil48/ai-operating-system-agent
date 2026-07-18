"""
Vector DB — backing store for the Knowledge Base (uploaded PDFs, docs, etc)
used by rag/ingest.py and rag/retrieve.py.

This implementation is intentionally lightweight: a persistent JSON store
plus fast hashed embeddings and cosine similarity. That keeps ingestion and
retrieval dependency-free, quick to load, and good enough for small-to-medium
knowledge bases without needing a separate database service.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.logging_config import get_logger
from backend.rag.text import embed_text, normalize_text

log = get_logger(__name__)


@dataclass(slots=True)
class VectorRecord:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


class VectorDB:
    def __init__(self, store_path: str | None = None, dimension: int | None = None):
        self._path = Path(store_path or settings.RAG_STORE_PATH).expanduser()
        self._dimension = dimension or settings.RAG_EMBED_DIM
        self._records: list[VectorRecord] = []
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                log.info("[vectordb] no existing store at %s", self._path)
                self._records = []
                return

            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                raw_records = payload.get("records", []) if isinstance(payload, dict) else []
                self._records = [
                    VectorRecord(
                        id=str(item.get("id") or uuid.uuid4().hex),
                        source=str(item.get("source") or "unknown"),
                        text=str(item.get("text") or ""),
                        metadata=dict(item.get("metadata") or {}),
                        embedding=[float(value) for value in item.get("embedding") or []],
                    )
                    for item in raw_records
                    if isinstance(item, dict) and item.get("text")
                ]
                log.info("[vectordb] loaded %d record(s) from %s", len(self._records), self._path)
            except Exception:
                log.exception("[vectordb] failed to load store from %s", self._path)
                self._records = []

    def _save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "dimension": self._dimension,
                "records": [
                    {
                        "id": record.id,
                        "source": record.source,
                        "text": record.text,
                        "metadata": record.metadata,
                        "embedding": record.embedding,
                    }
                    for record in self._records
                ],
            }
            self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids: list[str] = []
        with self._lock:
            for index, raw_text in enumerate(texts):
                text = normalize_text(raw_text)
                if not text:
                    continue

                metadata = dict(metadatas[index]) if metadatas and index < len(metadatas) else {}
                source = str(metadata.get("source") or "unknown")
                record = VectorRecord(
                    id=str(metadata.get("id") or uuid.uuid4().hex),
                    source=source,
                    text=text,
                    metadata=metadata,
                    embedding=embed_text(text, self._dimension),
                )
                self._records.append(record)
                ids.append(record.id)

            if ids:
                self._save()
        return ids

    def query(self, text: str, k: int = 4) -> list[dict[str, Any]]:
        if not self._records:
            return []

        query_embedding = embed_text(text, self._dimension)
        scored: list[tuple[float, VectorRecord]] = []

        for record in self._records:
            if not record.embedding:
                continue
            score = sum(left * right for left, right in zip(query_embedding, record.embedding))
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, record in scored[: max(1, k)]:
            results.append(
                {
                    "id": record.id,
                    "source": record.source,
                    "text": record.text,
                    "score": round(score, 4),
                    "metadata": record.metadata,
                }
            )
        return results

    def stats(self) -> dict[str, Any]:
        sources = sorted({record.source for record in self._records})
        return {
            "documents": len(self._records),
            "sources": sources,
            "store_path": str(self._path),
        }

    def clear(self) -> None:
        with self._lock:
            self._records = []
            if self._path.exists():
                self._path.unlink()


vector_db = VectorDB()
