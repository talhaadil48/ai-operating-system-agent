"""
Single point of truth for "which brain is the agent using".

Fallback chain (automatic, no restart needed):
  1. GROQ_API_KEY_1  (or GROQ_API_KEY for backward-compat)
  2. GROQ_API_KEY_2
  3. GROQ_API_KEY_3
  4. Gemini (GEMINI_API_KEY)  ← last resort

Any key that throws a rate-limit / auth / connection error is silently
skipped and the next candidate is tried immediately — no sleep, no delay.

Add a new provider (OpenAI, vLLM, etc.) by:
  1. Writing a class in backend/llm/ that implements BaseLLMProvider
  2. Inserting it into get_llm() below
  3. Adding its key to .env

Nothing in agent/, tools/, or api/ needs to change.
"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.llm.ollama_llm import OllamaProvider
from backend.llm.groq_llm import GroqProvider
from backend.llm.gemini_llm import GeminiProvider
from backend.logging_config import get_logger

log = get_logger(__name__)

# ── Rate-limit / quota error detection ────────────────────────────────────────
# These substrings appear in Groq's 429 / quota-exceeded error messages.
_GROQ_QUOTA_SIGNALS: tuple[str, ...] = (
    "rate_limit_exceeded",
    "rate limit",
    "429",
    "quota",
    "too many requests",
    "x-ratelimit",
    "tokens per",
    "requests per",
)


def _is_quota_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a Groq rate-limit / quota error."""
    msg = str(exc).lower()
    return any(sig in msg for sig in _GROQ_QUOTA_SIGNALS)


# ── Wrapped model that falls back on quota errors ─────────────────────────────

class _FallbackLLM:
    """
    Thin wrapper that holds an ordered list of LangChain chat models.

    On every call it tries models[0] first; if that raises a quota error it
    logs a warning and immediately retries with models[1], then models[2], …

    All other exceptions are re-raised immediately so real bugs surface fast.

    The wrapper exposes .invoke(), .bind_tools(), and .stream() so it is a
    drop-in replacement for any LangChain chat model.
    """

    def __init__(self, models: list[Any], labels: list[str]):
        if not models:
            raise RuntimeError("_FallbackLLM: no models provided — check your .env keys.")
        self._models = models
        self._labels = labels
        # bind_tools returns a new wrapper; track bound state per model.
        self._bound: list[Any] = list(models)

    # --- internal --------------------------------------------------------

    def _call(self, method: str, *args, **kwargs):
        last_exc: Exception | None = None
        for idx, model in enumerate(self._bound):
            label = self._labels[idx] if idx < len(self._labels) else f"model[{idx}]"
            try:
                return getattr(model, method)(*args, **kwargs)
            except Exception as exc:
                log.warning(
                    "[fallback-llm] %s failed during '%s' -> trying next backend immediately. "
                    "Error: %s",
                    label, method, exc,
                )
                last_exc = exc
                continue
        log.error(
            "[fallback-llm] All %d model(s) failed. Last error: %s",
            len(self._bound), last_exc,
        )
        raise RuntimeError(
            f"All LLM backends are unavailable (unreachable or quota exceeded). "
            f"Last error: {last_exc}"
        ) from last_exc

    # --- public LangChain interface --------------------------------------

    def invoke(self, *args, **kwargs):
        return self._call("invoke", *args, **kwargs)

    def stream(self, *args, **kwargs):
        # stream() is a generator; we try each backend and yield from the first
        # that succeeds.
        last_exc: Exception | None = None
        for idx, model in enumerate(self._bound):
            label = self._labels[idx] if idx < len(self._labels) else f"model[{idx}]"
            try:
                yield from getattr(model, "stream")(*args, **kwargs)
                return
            except Exception as exc:
                log.warning(
                    "[fallback-llm] %s failed during stream -> trying next backend immediately. "
                    "Error: %s",
                    label, exc,
                )
                last_exc = exc
                continue
        raise RuntimeError(
            f"All LLM backends unavailable (stream). Last error: {last_exc}"
        ) from last_exc

    def bind_tools(self, tools, **kwargs):
        """Return a new _FallbackLLM where every backend has tools bound."""
        bound_models = []
        bound_labels = []
        for idx, model in enumerate(self._models):
            label = self._labels[idx] if idx < len(self._labels) else f"model[{idx}]"
            try:
                bound_models.append(model.bind_tools(tools, **kwargs))
                bound_labels.append(label)
            except Exception as exc:
                log.warning(
                    "[fallback-llm] Could not bind tools to %s (will skip): %s",
                    label, exc,
                )
        if not bound_models:
            raise RuntimeError("No backend supports bind_tools — check provider compatibility.")
        wrapper = _FallbackLLM(bound_models, bound_labels)
        # Keep originals so re-binding works correctly.
        wrapper._models = self._models
        return wrapper

    # passthrough for anything else the agent graph accesses
    def __getattr__(self, name: str):
        return getattr(self._bound[0], name)


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm():
    """
    Build and return the LLM used by the agent graph.

    When LLM_PROVIDER=groq  → returns a _FallbackLLM chain:
        Groq key 1 → Groq key 2 → Groq key 3 → Gemini
    When LLM_PROVIDER=ollama → returns the Ollama model directly (no fallback).
    """
    provider = settings.LLM_PROVIDER.strip().lower()
    log.info("Selecting LLM provider: LLM_PROVIDER=%r", provider)

    # ── Ollama (no fallback needed — local model) ──────────────────────────
    if provider == "ollama":
        model = OllamaProvider().get_model()
        log.info("LLM ready: provider=ollama model=%r", model)
        return model

    # ── Groq with multi-key fallback → Gemini ─────────────────────────────
    if provider == "groq":
        models: list[Any] = []
        labels: list[str] = []

        groq_keys = settings.groq_api_keys
        if not groq_keys:
            log.warning("No Groq API keys configured. Will fall through to Gemini.")

        for i, key in enumerate(groq_keys, start=1):
            label = f"Groq-key-{i}"
            try:
                model = GroqProvider(api_key=key).get_model()
                models.append(model)
                labels.append(label)
                log.info("Registered fallback backend: %s", label)
            except Exception as exc:
                log.warning("Could not initialise %s (skipping): %s", label, exc)

        # Gemini as last resort
        if settings.GEMINI_API_KEY:
            try:
                gemini_model = GeminiProvider().get_model()
                models.append(gemini_model)
                labels.append("Gemini")
                log.info("Registered fallback backend: Gemini (last resort)")
            except Exception as exc:
                log.warning("Could not initialise Gemini fallback: %s", exc)
        else:
            log.warning(
                "GEMINI_API_KEY not set — Gemini fallback unavailable. "
                "Add it to .env for full resilience."
            )

        if not models:
            raise RuntimeError(
                "No LLM backends could be initialised. "
                "Set at least one of GROQ_API_KEY_1 or GEMINI_API_KEY in .env."
            )

        if len(models) == 1:
            log.info("Only one backend available — no fallback chain needed.")
            return models[0]

        log.info(
            "LLM fallback chain ready: %s",
            " → ".join(labels),
        )
        return _FallbackLLM(models, labels)

    # ── Unknown provider ───────────────────────────────────────────────────
    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        f"Supported: ollama, groq"
    )