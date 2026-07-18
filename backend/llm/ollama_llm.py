"""
Ollama provider. Uses langchain-ollama's ChatOllama, which supports
tool-calling (bind_tools) for models like qwen3 that support it.

To change the model, just edit OLLAMA_MODEL in your .env file —
nothing here needs to change.
"""

from langchain_ollama import ChatOllama

from backend.config import settings
from backend.llm.base import BaseLLMProvider
from backend.logging_config import get_logger

log = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = None, base_url: str = None, temperature: float = None):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        log.debug(
            "OllamaProvider configured: model=%s base_url=%s temperature=%s",
            self.model, self.base_url, self.temperature,
        )

    def get_model(self):
        log.info(
            "Connecting to Ollama at %s (model=%s, temperature=%s). "
            "If this hangs or errors, make sure `ollama serve` is running "
            "and the model has been pulled (`ollama pull %s`).",
            self.base_url, self.model, self.temperature, self.model,
        )
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )
