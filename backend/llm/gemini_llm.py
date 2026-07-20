"""
Google Gemini provider via langchain-google-genai.

This is used as the LAST-RESORT fallback when every configured Groq API key
has hit its rate limit or returned an error.

To enable:
  - Set GEMINI_API_KEY in .env
  - Optionally set GEMINI_MODEL (default: gemini-2.0-flash)
"""

from backend.config import settings
from backend.llm.base import BaseLLMProvider
from backend.logging_config import get_logger

log = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model: str = None, api_key: str = None, temperature: float = None):
        self.model = model or settings.GEMINI_MODEL
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.temperature = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
        log.debug(
            "GeminiProvider configured: model=%s temperature=%s api_key_set=%s",
            self.model, self.temperature, bool(self.api_key),
        )

    def get_model(self):
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env to use Gemini as fallback."
            )

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from e

        log.info(
            "Connecting to Gemini (model=%s, temperature=%s).",
            self.model, self.temperature,
        )
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_retries=0,
        )
