"""
Groq provider via langchain-groq ChatGroq.

To use:
  - Set LLM_PROVIDER=groq
  - Set GROQ_API_KEY
  - Choose GROQ_MODEL in .env (e.g. llama-3.1-8b-instant, llama-3.3-70b-versatile)
"""

from langchain_groq import ChatGroq

from backend.config import settings
from backend.llm.base import BaseLLMProvider
from backend.logging_config import get_logger

log = get_logger(__name__)


class GroqProvider(BaseLLMProvider):
    def __init__(self, model: str = None, api_key: str = None, temperature: float = None):
        self.model = model or settings.GROQ_MODEL
        self.api_key = api_key or settings.GROQ_API_KEY
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        log.debug(
            "GroqProvider configured: model=%s temperature=%s api_key_set=%s",
            self.model, self.temperature, bool(self.api_key),
        )

    def get_model(self):
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env when LLM_PROVIDER=groq."
            )

        log.info(
            "Connecting to Groq (model=%s, temperature=%s).",
            self.model, self.temperature,
        )
        return ChatGroq(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_retries=0,
        )