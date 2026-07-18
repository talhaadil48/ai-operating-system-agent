# AI OS — Single-Agent Architecture (LangGraph + Ollama)

Minimal, modular scaffold for a single-agent AI OS. One LLM (the "brain"),
core tools, and a clean structure to add more tools, persistent memory, and
RAG without rewriting everything.

## Structure

```
backend/
  agent/
    graph.py       # the LangGraph graph (this IS the single agent)
    state.py        # graph state (message list)
    router.py       # decides if/which tool to call
    executor.py      # runs the chosen tool
    planner.py       # stub for future multi-step planning
  llm/
    factory.py       # get_llm() — swap models/providers from .env only
    ollama_llm.py     # Ollama implementation (qwen3:8b by default)
    base.py           # interface every provider must follow
  tools/
    registry.py       # ALL_TOOLS — add new tools here
    calculator.py      # the one working tool
    base.py            # convention/docs for writing new tools
  memory/
    conversation.py    # short-term per-session chat memory (implemented)
    long_term.py        # user facts memory (stub)
    vectordb.py          # persistent knowledge base backing store
  rag/
    ingest.py            # document chunking + ingestion
    retrieve.py           # context retrieval for answers/tools
    text.py              # extraction, chunking, embeddings helpers
  api/
    main.py                # FastAPI /chat endpoint
  config.py                 # all settings, loaded from .env
  logging_config.py          # central logging setup — see "Debugging" below
main.py                      # CLI chat loop (fastest way to test)
requirements.txt
.env.example
```

## Setup

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen3:8b
   ollama serve
   ```

2. Python 3.11 virtual env:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Config:
   ```bash
   cp .env.example .env
   # edit .env if you want a different model/host
   ```

## Run

CLI (fastest way to test the loop):
```bash
python main.py
```
Try: `what is 234 * 17?` — you should see it route through the calculator tool.

API server:
```bash
uvicorn backend.api.main:app --reload
```
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "what is 12 * 7?"}'
```

### RAG / document upload

Upload plain text, markdown, JSON, CSV, HTML, or DOCX files and search them
through the local knowledge base:

```bash
curl -X POST http://localhost:8000/rag/upload-text \
  -H "Content-Type: application/json" \
  -d '{"source": "notes.md", "text": "hello world"}'
```

```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the document say about deployment?", "k": 4}'
```

For file uploads:

```bash
curl -X POST http://localhost:8000/rag/upload-file \
  -F "file=@./mydoc.md" \
  -F "source=mydoc.md"
```

The knowledge base is stored locally at `.ai_os/rag_store.json` by default.

## Swapping the LLM

Everything goes through `backend/llm/factory.py`. To change model, just
edit `.env`:
```
OLLAMA_MODEL=llama3.1:8b
```
To add a whole new provider (OpenAI, Groq, vLLM...):
1. Create `backend/llm/your_provider.py` implementing `BaseLLMProvider`
   (see `backend/llm/base.py`)
2. Register it in `PROVIDERS` in `backend/llm/factory.py`
3. Set `LLM_PROVIDER=your_provider` in `.env`

No other file needs to change — the agent graph only ever calls `get_llm()`.

## Adding a new tool

1. Write it in `backend/tools/your_tool.py` as a `@tool`-decorated
   function with a clear docstring (see `backend/tools/base.py`)
2. Import + add it to `ALL_TOOLS` in `backend/tools/registry.py`

That's it — the graph, router, and executor all pick it up automatically.

## Debugging

The whole agent — every layer — now logs what it's doing through one
central setup in `backend/logging_config.py`. No extra dependencies,
just Python's stdlib `logging`, wired up so it's actually useful.

**Turn it on:**
```bash
# .env
DEBUG=true
```
or for a single run without touching `.env`:
```bash
DEBUG=true python main.py
DEBUG=true uvicorn backend.api.main:app --reload
```

**What you get with `DEBUG=true`:**
- Every message sent to the LLM (the full context stack, roles + content)
- Every LLM call's timing (`llm_invoke ... done in 812.3ms`)
- Whether the LLM decided to call a tool or answer directly, and why
  (the raw tool name + args it requested)
- Every tool execution: name, args in, result/error out, timing
- Every routing decision (`tools` vs `END`)
- Memory reads/writes (`get`/`append`/`clear`) per session, with message counts
- Full request/response logging on the API (method, path, status, timing)
- Full tracebacks on any unhandled exception, in the agent, a tool, or the API

**Without `DEBUG=true`** (default `LOG_LEVEL=INFO`) you still get a clean,
one-line-per-step trace of the important stuff — LLM calls started/decided,
tools called, requests in/out — just without full prompt/message dumps.

**Reading the logs — every line is tagged with a `turn=xxxxxxxx` id:**
```
14:32:01 INFO     turn=a1b2c3d4 api.main   New chat turn. session=test message='what is 12 * 7?'
14:32:01 INFO     turn=a1b2c3d4 agent.graph [agent] invoking LLM with 2 message(s) in context
14:32:02 INFO     turn=a1b2c3d4 agent.graph [agent] LLM requested 1 tool call(s): calculator({'expression': '12 * 7'})
14:32:02 DEBUG    turn=a1b2c3d4 agent.router [router] decision=tools (routing to tool executor)
14:32:02 INFO     turn=a1b2c3d4 agent.executor [tools] executing calculator({'expression': '12 * 7'}) [id=call_1]
14:32:02 INFO     turn=a1b2c3d4 agent.executor [tools] result <- calculator (status=success): 84
14:32:02 INFO     turn=a1b2c3d4 agent.graph [agent] LLM answered directly (no tool call). reply='12 * 7 is 84.'
14:32:02 INFO     turn=a1b2c3d4 api.main   Turn a1b2c3d4 complete. session=test reply='12 * 7 is 84.'
```
Every log line for one conversation turn shares the same `turn=` id (across
the API/CLI, agent, router, and tool executor), so `grep turn=a1b2c3d4` in
your terminal shows the entire journey of that one message end-to-end —
even if multiple requests are interleaved (e.g. concurrent API calls).

**Other log settings** (in `.env` or as env vars):
- `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` — set the level directly instead of
  using `DEBUG=true`/`false`
- `LOG_JSON=true` — emit one JSON object per line instead of colored text,
  if you want to pipe logs into something like Loki/ELK/Datadog

**Common issues this surfaces immediately:**
- Ollama not running / model not pulled → connection error is logged right
  where `ChatOllama` is constructed, with a hint to run `ollama serve` /
  `ollama pull <model>`
- Agent stuck in a tool-call loop → you'll see repeated
  `[agent] LLM requested ... tool call(s)` lines for the same turn
- Tool silently failing → `[tools] result ... (status=error): ...` is
  logged at ERROR level instead of blending in
- Memory not persisting across turns → `[memory] get/append/clear` lines
  show exactly how many messages are in a session at each point
