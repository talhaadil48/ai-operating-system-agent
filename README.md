# AI Operating System — Single-Agent AI (LangGraph + Groq + PostgreSQL)

A modular, production-ready **single-agent AI OS** built on LangGraph. One LLM brain, 27 tools, persistent long-term memory backed by PostgreSQL, RAG knowledge base, multi-key LLM fallback chains, and a clean FastAPI + CLI interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Single Agent** | LangGraph-powered loop: LLM → Router → Tool Executor → LLM |
| 🔧 **27 Tools** | Calculator, Web Search, File Manager, Terminal, Screen Analyzer, Process Manager, Weather, and more |
| 💾 **Long-Term Memory** | PostgreSQL-backed persistent user preferences via SQLAlchemy |
| 📚 **RAG** | Upload documents (PDF, MD, TXT, CSV, DOCX) and query them via a local vector store |
| 🔄 **LLM Fallback Chain** | Groq Key 1 → Key 2 → Key 3 → Gemini (automatic failover) |
| 🖥️ **Dual Interface** | CLI (for quick testing) + FastAPI REST server |
| 🪲 **Structured Logging** | Every turn tagged with a unique `turn=` ID for end-to-end tracing |

---

## 📁 Project Structure

```
ai-os/
├── main.py                        # CLI chat loop
├── requirements.txt
├── .env.example                   # ← copy this to .env
│
└── backend/
    ├── config.py                  # All settings loaded from .env
    ├── logging_config.py          # Centralized structured logging
    │
    ├── agent/
    │   ├── graph.py               # LangGraph agent graph (the brain)
    │   ├── state.py               # Graph state (message list)
    │   ├── router.py              # Decides tool vs direct answer
    │   └── executor.py            # Runs tool calls
    │
    ├── llm/
    │   ├── factory.py             # get_llm() — swap providers via .env
    │   ├── groq_llm.py            # Groq (multi-key fallback)
    │   ├── ollama_llm.py          # Ollama (local models)
    │   ├── gemini_llm.py          # Gemini (last-resort fallback)
    │   └── base.py                # Provider interface
    │
    ├── tools/
    │   ├── registry.py            # ALL_TOOLS list — add tools here
    │   ├── calculator.py          # Arithmetic
    │   ├── web_search.py          # Web search via Serper API
    │   ├── file_manager.py        # Read/write/copy/move/delete/list files
    │   ├── terminal.py            # Run shell commands, kill processes
    │   ├── process_manager.py     # List/start/stop processes, resource usage
    │   ├── app_launcher.py        # Open apps, folders, URLs
    │   ├── screen_analyzer.py     # Screenshot + vision LLM analysis
    │   ├── web_utilities.py       # Scrape, summarize, check websites, weather
    │   ├── knowledge_base.py      # RAG search over uploaded documents
    │   ├── search_workspace.py    # Search project files by text
    │   ├── system_status.py       # OS/Python/LLM status
    │   └── memory_tool.py         # Save/recall/delete long-term memories
    │
    ├── memory/
    │   ├── conversation.py        # Short-term session memory (sliding window)
    │   ├── long_term.py           # Long-term PostgreSQL memory (SQLAlchemy)
    │   └── vectordb.py            # Local JSON vector store for RAG
    │
    ├── database/
    │   ├── connection.py          # SQLAlchemy engine + session factory
    │   └── models.py              # LongTermMemoryRecord table model
    │
    ├── rag/
    │   ├── ingest.py              # Document chunking + ingestion
    │   ├── retrieve.py            # Context retrieval
    │   └── text.py                # Extraction, chunking, embeddings
    │
    └── api/
        └── main.py                # FastAPI /chat + /rag endpoints
```

---

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-os.git
cd ai-os
```

### 2. Create a Python Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys and database URI
```

See the [Environment Variables](#-environment-variables) section below for a full reference.

### 5. Set Up PostgreSQL (Long-Term Memory)

Create a PostgreSQL database locally or use a cloud provider (e.g. [Neon](https://neon.tech)):

```sql
CREATE DATABASE ai_os;
```

Add your connection string to `.env`:
```env
POSTGRES_URI=postgresql://postgres:password@localhost:5432/ai_os
```

The `long_term_memories` table is created **automatically** on first startup.

---

## ▶️ Running

### CLI (Fastest way to test)

```bash
python main.py
```

```
AI OS — type 'exit' to quit.
You: what is 234 * 17?
AI: 234 * 17 = 3,978
```

### API Server

```bash
uvicorn backend.api.main:app --reload
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "message": "what is the weather in Lahore?"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## 📋 Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | ✅ | `groq` | LLM backend: `groq`, `ollama`, or `gemini` |
| `GROQ_API_KEY_1` | ✅ | — | Primary Groq API key |
| `GROQ_API_KEY_2` | ➖ | — | Backup Groq API key (optional) |
| `GROQ_API_KEY_3` | ➖ | — | Backup Groq API key (optional) |
| `GROQ_MODEL` | ➖ | `llama-3.3-70b-versatile` | Groq model to use |
| `GROQ_TEMPERATURE` | ➖ | `0.3` | LLM temperature |
| `GEMINI_API_KEY` | ➖ | — | Google Gemini API key (last-resort fallback) |
| `GEMINI_MODEL` | ➖ | `gemini-flash-latest` | Gemini model |
| `OLLAMA_MODEL` | ➖ | `qwen3:8b` | Ollama local model name |
| `OLLAMA_BASE_URL` | ➖ | `http://localhost:11434` | Ollama server URL |
| `POSTGRES_URI` | ✅ | — | PostgreSQL connection string for long-term memory |
| `SERPER_API_KEY` | ➖ | — | Serper.dev API key for web search |
| `GROQ_VISION_MODEL` | ➖ | `qwen/qwen3.6-27b` | Vision model for screen analysis tool |
| `API_HOST` | ➖ | `0.0.0.0` | FastAPI server host |
| `API_PORT` | ➖ | `8000` | FastAPI server port |
| `DEBUG` | ➖ | `false` | Enable verbose debug logging |
| `LOG_LEVEL` | ➖ | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_JSON` | ➖ | `false` | Emit JSON logs (for Loki/ELK/Datadog) |
| `MEMORY_MAX_TURNS` | ➖ | `1` | Max conversation turns kept in context window |
| `LTM_MAX_FACTS` | ➖ | `5` | Max long-term memory facts injected per prompt |

---

## 🧠 Long-Term Memory

The agent can **remember user preferences and facts across sessions** using PostgreSQL.

**How it works:**
- When you tell the agent a persistent fact (e.g. `"My name is Talha"`, `"I prefer FastAPI"`), it calls `save_long_term_memory` and stores it in PostgreSQL.
- On every new turn, saved facts are recalled from PostgreSQL and injected into the system prompt automatically.
- Memories survive server restarts and session resets.

**Example:**
```
You: Remember that I prefer building backends with FastAPI and async Python.
AI: Got it! I'll remember that you prefer FastAPI with async Python.

[Server restart]

You: What are my backend preferences?
AI: You prefer building backends with FastAPI and async Python.
```

**Memory categories:** `preference`, `environment`, `biography`, `rule`

**Debug log when saving:**
```
========================================================================
[DEBUG LOG] [LTM] SAVING TO POSTGRESQL DATABASE:
  User ID  : default
  Category : preference
  Fact     : Prefers FastAPI with async Python
========================================================================
[long_term_memory] [DEBUG] SUCCESS: Saved memory record to PostgreSQL!
```

---

## 📚 RAG — Knowledge Base

Upload documents and query them through the agent's `knowledge_base_search` tool.

**Upload text:**
```bash
curl -X POST http://localhost:8000/rag/upload-text \
  -H "Content-Type: application/json" \
  -d '{"source": "my-notes.md", "text": "The deployment runs on port 8080..."}'
```

**Upload a file (PDF, MD, TXT, DOCX, CSV):**
```bash
curl -X POST http://localhost:8000/rag/upload-file \
  -F "file=@./docs/architecture.pdf" \
  -F "source=architecture.pdf"
```

**Search directly:**
```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what port does deployment run on?", "k": 4}'
```

**Stats:**
```bash
curl http://localhost:8000/rag/stats
```

The knowledge base is stored locally at `.ai_os/rag_store.json`.

---

## 🔧 Adding a New Tool

1. Create `backend/tools/my_tool.py`:
```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Description of what this tool does and when to use it."""
    return f"Result: {input}"
```

2. Register it in `backend/tools/registry.py`:
```python
from backend.tools.my_tool import my_tool

ALL_TOOLS = [
    ...
    my_tool,
]
```

The agent graph, router, and executor pick it up automatically.

---

## 🔄 Swapping the LLM Provider

Edit `.env` only — no code changes needed:

```env
# Use Ollama locally
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b

# Use Groq (recommended)
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.3-70b-versatile

# Use Gemini
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
```

To add a new provider: implement `BaseLLMProvider` in `backend/llm/` and register it in `factory.py`.

---

## 🪲 Debugging

Enable verbose logging:
```env
# .env
DEBUG=true
```

Or for a single run:
```bash
DEBUG=true python main.py
```

Every log line is tagged with a `turn=` ID:
```
17:32:01 INFO  turn=a1b2c3d4  agent.graph    [agent] invoking LLM with 3 message(s) in context
17:32:02 INFO  turn=a1b2c3d4  agent.graph    [agent] LLM requested 1 tool call(s): web_search({'query': '...'})
17:32:03 INFO  turn=a1b2c3d4  agent.executor [tools] result <- web_search (status=success): ...
17:32:04 INFO  turn=a1b2c3d4  agent.graph    [agent] LLM answered directly. reply='...'
```

Use `grep turn=a1b2c3d4` to trace one full request end-to-end across all layers.

---

## 📦 Requirements

- Python 3.11+
- PostgreSQL (local or cloud e.g. [Neon](https://neon.tech))
- Groq API key → [console.groq.com](https://console.groq.com) (free tier available)
- Serper API key → [serper.dev](https://serper.dev) (for web search, optional)
- Gemini API key → [aistudio.google.com](https://aistudio.google.com) (optional, fallback only)

---

## 📄 License

MIT
