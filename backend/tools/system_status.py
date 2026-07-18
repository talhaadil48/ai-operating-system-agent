"""System status tool."""

import os
import platform
import sys
from langchain_core.tools import tool
from backend.config import settings

@tool
def system_status(query: str = "") -> str:
    """Get the current system status, environment variables, working directory, and operating system info.
    Use this when the user asks about the environment, status, or current directory."""
    cwd = os.getcwd()
    system_info = f"OS: {platform.system()} {platform.release()}\n"
    system_info += f"Python: {sys.version}\n"
    system_info += f"Current Working Directory: {cwd}\n"
    system_info += f"LLM Provider: {settings.LLM_PROVIDER}\n"
    if settings.LLM_PROVIDER == "ollama":
        system_info += f"Ollama Model: {settings.OLLAMA_MODEL}\n"
        system_info += f"Ollama Base URL: {settings.OLLAMA_BASE_URL}\n"
    elif settings.LLM_PROVIDER == "groq":
        system_info += f"Groq Model: {settings.GROQ_MODEL}\n"
    
    system_info += f"RAG Store Path: {settings.RAG_STORE_PATH}\n"
    return system_info
