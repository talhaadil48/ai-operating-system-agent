"""
RAG retrieval — pulls relevant chunks from the Knowledge Base for a query.

Returns compact citations from the persistent vector store so the agent can
ground answers in uploaded documents quickly.
"""

from backend.config import settings
from backend.memory.vectordb import vector_db


def retrieve_context(query: str, k: int = 4) -> list[str]:
    results = vector_db.query(query, k=max(1, k or settings.RAG_TOP_K))
    context: list[str] = []
    for index, result in enumerate(results, start=1):
        text = str(result.get("text") or "").strip()
        source = str(result.get("source") or "unknown")
        score = result.get("score", 0.0)
        context.append(
            f"[{index}] source={source} score={score}: {text}"
        )
    return context
