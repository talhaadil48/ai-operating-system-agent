"""
Central config for the whole AI OS.

Everything that could change between environments/models lives here,
loaded from environment variables (see .env.example).

To swap the LLM: change LLM_PROVIDER / model vars in your .env file.
No other code needs to change.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

    # Groq — supports up to 3 keys with automatic fallback.
    # GROQ_API_KEY_1 is tried first; 2 and 3 are backups.
    # GROQ_API_KEY is accepted as an alias for key 1.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # legacy / alias
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2", "")
    GROQ_API_KEY_3: str = os.getenv("GROQ_API_KEY_3", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

    # Gemini — last-resort fallback when all Groq keys are exhausted.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # RAG / Knowledge Base
    RAG_STORE_PATH: str = os.getenv("RAG_STORE_PATH", ".ai_os/rag_store.json")
    RAG_EMBED_DIM: int = int(os.getenv("RAG_EMBED_DIM", "512"))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "900"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    RAG_MAX_UPLOAD_BYTES: int = int(os.getenv("RAG_MAX_UPLOAD_BYTES", "15000000"))

    # Conversation Memory
    MEMORY_MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "3"))
    MEMORY_MAX_SUMMARY_WORDS: int = int(os.getenv("MEMORY_MAX_SUMMARY_WORDS", "250"))

    # Debugging / logging (see backend/logging_config.py)
    # DEBUG=true turns on verbose logging everywhere: full prompts sent to
    # the LLM, raw tool args/results, routing decisions, timings, etc.
    DEBUG: bool = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
    LOG_JSON: bool = os.getenv("LOG_JSON", "false").strip().lower() in {"1", "true", "yes", "on"}

    # Database & Long-Term Memory (LTM)
    POSTGRES_URI: str = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/ai_os")
    LTM_MAX_FACTS: int = int(os.getenv("LTM_MAX_FACTS", "10"))
    LTM_MAX_SUMMARY_WORDS: int = int(os.getenv("LTM_MAX_SUMMARY_WORDS", "300"))



    # SERPER API KEY
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    @property
    def groq_api_keys(self) -> list[str]:
        """Return the list of configured Groq API keys (non-empty only)."""
        candidates = [self.GROQ_API_KEY_1, self.GROQ_API_KEY_2, self.GROQ_API_KEY_3]
        return [k.strip() for k in candidates if k and k.strip()]


settings = Settings()