"""
Single point of truth for "which brain is the agent using".

Add a new provider (OpenAI, Groq, vLLM, whatever) by:
  1. Writing a new class in backend/llm/ that implements BaseLLMProvider
  2. Registering it in PROVIDERS below
  3. Setting LLM_PROVIDER in .env

Nothing in agent/, tools/, or api/ needs to change.
"""

from backend.config import settings
from backend.llm.ollama_llm import OllamaProvider
from backend.llm.groq_llm import GroqProvider
from backend.logging_config import get_logger

log = get_logger(__name__)

PROVIDERS = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    # "openai": OpenAIProvider,   # <- example of how you'd add another one later
}


def get_llm():
    log.info("Selecting LLM provider: LLM_PROVIDER=%r", settings.LLM_PROVIDER)
    provider_cls = PROVIDERS.get(settings.LLM_PROVIDER)
    if provider_cls is None:
        log.error(
            "Unknown LLM_PROVIDER %r. Available providers: %s",
            settings.LLM_PROVIDER, list(PROVIDERS.keys()),
        )
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
            f"Available: {list(PROVIDERS.keys())}"
        )
    provider = provider_cls()
    model = provider.get_model()
    log.info("LLM ready: provider=%s model=%r", settings.LLM_PROVIDER, model)
    return model