"""
Every LLM backend must return a LangChain-compatible chat model
(something with .invoke(), .stream(), .bind_tools() etc).

This is the ONLY contract the rest of the app relies on, which is
what makes the model swappable — the agent graph never imports
Ollama directly, it always goes through llm/factory.py.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def get_model(self):
        """Return a LangChain chat model instance."""
        raise NotImplementedError
