"""Knowledge base search tool for uploaded documents and indexed notes."""

from langchain_core.tools import tool

from backend.config import settings
from backend.rag.retrieve import retrieve_context


@tool
def knowledge_base_search(query: str, k: int = 4) -> str:
    """Search uploaded documents and indexed knowledge base content.

    Use this when the user asks about uploaded files, indexed documents, or
    knowledge stored in the AI OS RAG store. The tool returns the most relevant
    snippets with source citations.
    """
    context = retrieve_context(query, k=k or settings.RAG_TOP_K)
    if not context:
        return "No indexed knowledge base documents were found. Upload a document first."
    return "\n\n".join(context)