"""
FastAPI wrapper around the agent graph.

    POST /chat  { "session_id": "abc", "message": "what is 12*7?" }
    -> { "session_id": "abc", "reply": "84" }

Run with:  uvicorn backend.api.main:app --reload
"""

import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.agent.graph import agent_graph
from backend.logging_config import get_logger, log_timing, new_turn
from backend.memory.conversation import conversation_memory
from backend.rag.ingest import ingest_document
from backend.rag.retrieve import retrieve_context
from backend.rag.text import extract_text_from_bytes
from backend.memory.vectordb import vector_db
from backend.config import settings

log = get_logger(__name__)

app = FastAPI(title="AI OS", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class RagUploadTextRequest(BaseModel):
    source: str
    text: str


class RagUploadResponse(BaseModel):
    source: str
    chunks_added: int
    chunk_ids: list[str]


class RagSearchRequest(BaseModel):
    query: str
    k: int = 4


class RagStatsResponse(BaseModel):
    documents: int
    sources: list[str]
    store_path: str


class RagSearchResponse(BaseModel):
    query: str
    results: list[str]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every HTTP request/response with timing, so you can see
    exactly what hit the API and how long it took — including requests
    that never reach a route handler (404s, bad bodies, etc)."""
    start = time.perf_counter()
    log.info(">> %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("<< %s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so an agent/tool crash returns a clean 500 with a
    traceback in the logs instead of a bare stack trace to the client."""
    log.exception("Unhandled error handling %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


@app.get("/health")
def health():
    log.debug("Health check")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with new_turn(session_id=req.session_id) as turn_id:
        log.info("New chat turn. session=%s message=%r", req.session_id, req.message)

        history = conversation_memory.get(req.session_id)
        user_message = HumanMessage(content=req.message)

        with log_timing(log, "agent_graph.invoke", level=20):  # INFO level
            result = agent_graph.invoke(
                {
                    "messages": history + [user_message],
                    "summary": conversation_memory.get_summary(req.session_id)
                }
            )
        new_messages = result["messages"]

        # Persist the full updated history (includes tool calls/results) for
        # this session so the next turn has full context.
        conversation_memory.clear(req.session_id)
        conversation_memory.append(req.session_id, new_messages)

        reply = new_messages[-1].content
        log.info("Turn %s complete. session=%s reply=%r", turn_id, req.session_id, reply[:200])
        return ChatResponse(session_id=req.session_id, reply=reply)


@app.post("/rag/upload-text", response_model=RagUploadResponse)
def upload_text(req: RagUploadTextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    result = ingest_document(req.text, req.source)
    return RagUploadResponse(**result)


@app.post("/rag/upload-file", response_model=RagUploadResponse)
async def upload_file(file: UploadFile = File(...), source: str | None = Form(None)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > settings.RAG_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="uploaded file is too large")

    file_source = source or file.filename or "uploaded-file"
    try:
        text = extract_text_from_bytes(data, file.filename or file_source, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="no text could be extracted from the file")

    result = ingest_document(text, file_source)
    return RagUploadResponse(**result)


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(req: RagSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    return RagSearchResponse(query=req.query, results=retrieve_context(req.query, k=req.k))


@app.get("/rag/stats", response_model=RagStatsResponse)
def rag_stats():
    stats = vector_db.stats()
    return RagStatsResponse(**stats)
