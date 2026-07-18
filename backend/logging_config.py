"""
Central logging setup for the whole AI OS.

Import `get_logger(__name__)` anywhere you need to log — everything is
configured once, from here, based on env vars (see .env.example):

    DEBUG=true        -> log level DEBUG everywhere (very verbose: full
                         prompts, tool args/results, raw LLM responses)
    LOG_LEVEL=INFO     -> override level directly (DEBUG/INFO/WARNING/...)
    LOG_JSON=false      -> set true to emit one JSON object per line
                         instead of colored human-readable text (useful
                         if you pipe logs into something like Loki/ELK)

Every request (API call) or CLI turn gets a short `turn_id` that's
attached to *every* log line produced while handling it, via a
contextvar — so you can `grep turn=ab12cd` and see the entire journey
of that one message: API in -> memory load -> agent -> LLM call ->
router decision -> tool execution -> LLM call -> API out.

Usage:
    from backend.logging_config import get_logger, new_turn, log_timing

    log = get_logger(__name__)

    def handle_request(...):
        with new_turn(session_id):          # tags all logs in this block
            log.info("handling request")
            with log_timing(log, "llm_call"):
                ...
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Optional

# ---------------------------------------------------------------------------
# Turn / request correlation
# ---------------------------------------------------------------------------

_turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="-")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")


def current_turn_id() -> str:
    return _turn_id_var.get()


@contextmanager
def new_turn(session_id: str = "-", turn_id: Optional[str] = None):
    """Open a new logging 'turn' — every log emitted inside this `with`
    block (in this coroutine/thread, including nested function calls)
    is tagged with the same short turn_id and session_id, so one
    conversation turn's logs can be traced across every module."""
    tid = turn_id or uuid.uuid4().hex[:8]
    turn_token = _turn_id_var.set(tid)
    session_token = _session_id_var.set(session_id)
    try:
        yield tid
    finally:
        _turn_id_var.reset(turn_token)
        _session_id_var.reset(session_token)


class _ContextFilter(logging.Filter):
    """Injects turn_id/session_id into every log record so the formatter
    can print them (and so JSON logs include them as real fields)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.turn_id = _turn_id_var.get()
        record.session_id = _session_id_var.get()
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_COLORS = {
    "DEBUG": "\033[2;37m",     # dim white
    "INFO": "\033[36m",        # cyan
    "WARNING": "\033[33m",     # yellow
    "ERROR": "\033[31m",       # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        turn = getattr(record, "turn_id", "-")
        name = record.name.replace("backend.", "")

        if self.use_color:
            color = _COLORS.get(level, "")
            line = (
                f"{_DIM}{ts}{_RESET} "
                f"{color}{level:<8}{_RESET} "
                f"{_BOLD}turn={turn}{_RESET} "
                f"{_DIM}{name}{_RESET}  "
                f"{record.getMessage()}"
            )
        else:
            line = f"{ts} {level:<8} turn={turn} {name}  {record.getMessage()}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "turn_id": getattr(record, "turn_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_configured = False


def setup_logging() -> None:
    """Configure the root logger exactly once. Safe to call repeatedly
    (e.g. from both main.py and backend/api/main.py) — subsequent calls
    are no-ops."""
    global _configured
    if _configured:
        return
    _configured = True

    debug = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    level_name = os.getenv("LOG_LEVEL", "DEBUG" if debug else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    use_json = os.getenv("LOG_JSON", "false").strip().lower() in {"1", "true", "yes", "on"}
    use_color = sys.stderr.isatty() and not use_json

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else ColorFormatter(use_color=use_color))
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Noisy third-party loggers stay quieter unless we're in DEBUG mode.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if debug else logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured (level=%s, json=%s, debug=%s)", level_name, use_json, debug
    )


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger. Calls setup_logging() automatically so
    every module can just do `log = get_logger(__name__)` at import time
    with no separate init step required."""
    setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

@contextmanager
def log_timing(logger: logging.Logger, label: str, level: int = logging.DEBUG, **fields):
    """Context manager that logs how long a block took.

        with log_timing(log, "llm_call", model="qwen3:8b"):
            response = llm.invoke(messages)

    Logs on entry (level DEBUG) and on exit with elapsed ms. If the
    block raises, logs the exception with elapsed time too, then
    re-raises — nothing is swallowed.
    """
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.log(level, "-> %s starting%s", label, f" ({extra})" if extra else "")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("x  %s failed after %.1fms", label, elapsed_ms)
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log(level, "<- %s done in %.1fms", label, elapsed_ms)
