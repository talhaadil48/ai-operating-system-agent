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

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

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


    #SERPER API KEY
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    # Computer Interaction Layer (Tools #19 & #20)
    # Vision model for screen analysis — must be a model with vision support.
    # Ollama options: llava, minicpm-v, moondream, bakllava
    # Groq option: llama-4-scout-17b (used automatically when LLM_PROVIDER=groq)
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llava")
    SCREENSHOT_DIR: str = os.getenv("SCREENSHOT_DIR", ".ai_os/screenshots")


settings = Settings()