"""
Screen Analyzer Tool  (Computer Interaction Layer — Tool #20)

Sends a captured screenshot to a vision-capable LLM and returns a rich
plain-text description of:
  • Visible text, errors, warnings
  • UI elements (buttons, forms, menus, dialogs)
  • Application layout / structure
  • Actionable suggestions if an error is detected

Vision backend selection (in order of priority)
-------------------------------------------------
1. Ollama  — LLaVA or any other vision model pulled locally.
             Set VISION_MODEL=llava (or minicpm-v, moondream, etc.) in .env.
             Requires: langchain-ollama already in requirements.
2. Groq    — groq's vision endpoint (qwen/qwen3.6-27b by default).
             Override with GROQ_VISION_MODEL= in .env.
             Automatically used when LLM_PROVIDER=groq and GROQ_API_KEY is set.
3. Fallback — If no vision model is available, returns the screenshot metadata
              and a clear message explaining how to enable vision.

Usage flow
----------
  User:  "Fix this error"
  Agent: 1. calls capture_screen()  → gets screenshot path
         2. calls analyze_screen(path=<path>, task="find and fix the error")
         → LLM reads the image, identifies the error, explains the fix

Design notes
------------
• The screenshot is base64-encoded and sent inline — no external upload.
• The `task` parameter lets the user focus the analysis
  (e.g. "find all buttons", "read the error message", "describe the layout").
• Works without any API key if you have Ollama + a vision model running.
"""

import base64
import os
import time
from pathlib import Path

from langchain_core.tools import tool

from backend.config import settings
from backend.logging_config import get_logger

log = get_logger(__name__)

# ── default vision models ────────────────────────────────────────────────────
_OLLAMA_VISION_DEFAULT = os.getenv("VISION_MODEL", "llava")
# Override with GROQ_VISION_MODEL= in .env — check https://console.groq.com/docs/models for current IDs
_GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

# ── system prompt for vision analysis ────────────────────────────────────────
_SYSTEM_PROMPT = """You are a computer vision assistant integrated into an AI Operating System.
Your job is to analyze screenshots and describe what you see clearly and usefully.

When analyzing a screenshot, always cover:
1. **Active application** — what program is open?
2. **Visible text** — reproduce any important text, error messages, warnings verbatim.
3. **UI elements** — list visible buttons, menus, forms, dialogs, tabs.
4. **Layout** — describe the overall structure of the screen.
5. **Problems detected** — call out errors, warnings, loading states, or broken elements.
6. **Suggested actions** — if the user asked to fix something, provide clear steps.

Be concise but thorough. Format your response with markdown headings."""


def _load_image_as_base64(path: str) -> tuple[str, str]:
    """Load an image file and return (base64_string, mime_type)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Screenshot not found: {path}")
    ext = p.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def _analyze_with_ollama(image_b64: str, mime: str, user_prompt: str) -> str:
    """Send image to Ollama vision model."""
    from langchain_ollama import ChatOllama  # type: ignore
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = _OLLAMA_VISION_DEFAULT
    log.info("[screen_analyzer] using Ollama vision model: %s", model_name)

    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
            },
        ]
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), message])
    return response.content


def _analyze_with_groq(image_b64: str, mime: str, user_prompt: str) -> str:
    """Send image to Groq vision endpoint."""
    try:
        from langchain_groq import ChatGroq  # type: ignore
    except ImportError:
        raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")

    from langchain_core.messages import HumanMessage, SystemMessage

    log.info("[screen_analyzer] using Groq vision model: %s", _GROQ_VISION_MODEL)

    llm = ChatGroq(
        model=_GROQ_VISION_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_tokens=4096,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
    )

    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), message])
    return response.content


def _auto_capture_if_needed() -> str:
    """
    Take a fresh screenshot and return its path.
    Used when analyze_screen() is called without an explicit path.
    """
    from backend.tools.screen_capture import capture_screen  # local import to avoid circular

    # capture_screen returns a formatted string — extract the path from it
    result_text = capture_screen.invoke({"save_path": ""})
    # Parse "Saved to   : <path>"
    for line in result_text.splitlines():
        if "Saved to" in line:
            path = line.split(":", 1)[1].strip()
            return path

    # Fallback — try .ai_os/screenshots/ for the latest file
    shots_dir = Path(".ai_os") / "screenshots"
    if shots_dir.exists():
        files = sorted(shots_dir.glob("screenshot_*.png"))
        if files:
            return str(files[-1])

    raise RuntimeError("Could not auto-capture screenshot and no path provided.")


@tool
def analyze_screen(path: str = "", task: str = "") -> str:
    """Analyze a screenshot using a vision-capable AI model.

    Understands UI elements, reads error messages, identifies buttons and forms,
    and suggests fixes. Use this when the user says things like:
    - 'what is on my screen?' / 'what do you see?'
    - 'fix this error' / 'read this error message'
    - 'what buttons are on this page?'
    - 'describe my screen' / 'what application is open?'

    If no path is given, a fresh screenshot is taken automatically.

    Args:
        path: Absolute or relative path to a screenshot PNG/JPG.
              If empty, capture_screen() is called automatically first.
        task: Specific thing to focus on, e.g. 'find the error message',
              'list all clickable buttons', 'describe the form fields'.
              If empty, performs a full general analysis.
    """
    log.info("[screen_analyzer] starting analysis. path=%r task=%r", path, task)
    t0 = time.perf_counter()

    # ── get/take screenshot ──────────────────────────────────────────────────
    if not path:
        log.info("[screen_analyzer] no path given — auto-capturing screen …")
        try:
            path = _auto_capture_if_needed()
            log.info("[screen_analyzer] auto-captured: %s", path)
        except Exception as exc:
            return (
                f"Could not take a screenshot automatically: {exc}\n\n"
                "Please call capture_screen() first, then pass the returned path "
                "to analyze_screen(path=<path>)."
            )

    # ── load image ───────────────────────────────────────────────────────────
    try:
        image_b64, mime = _load_image_as_base64(path)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        log.error("[screen_analyzer] failed to load image: %s", exc)
        return f"Failed to load image at '{path}': {exc}"

    file_size_kb = Path(path).stat().st_size / 1024

    # ── build prompt ─────────────────────────────────────────────────────────
    user_prompt = (
        f"Please analyze this screenshot.\n\n"
        f"Specific task: {task if task else 'Perform a full general analysis of the screen.'}\n\n"
        f"Screenshot file: {path} ({file_size_kb:.1f} KB)"
    )

    # ── choose vision backend ────────────────────────────────────────────────
    analysis = ""
    backend_used = ""
    errors = []

    # 1. Try Groq if configured (fast, good quality)
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        try:
            analysis = _analyze_with_groq(image_b64, mime, user_prompt)
            backend_used = f"Groq ({_GROQ_VISION_MODEL})"
        except Exception as exc:
            log.warning("[screen_analyzer] Groq vision failed: %s", exc)
            errors.append(f"Groq: {exc}")

    # 2. Try Ollama vision model as primary or fallback
    if not analysis:
        try:
            analysis = _analyze_with_ollama(image_b64, mime, user_prompt)
            backend_used = f"Ollama ({_OLLAMA_VISION_DEFAULT})"
        except Exception as exc:
            log.warning("[screen_analyzer] Ollama vision failed: %s", exc)
            errors.append(f"Ollama ({_OLLAMA_VISION_DEFAULT}): {exc}")

    # 3. No vision model available
    if not analysis:
        setup_guide = (
            "## Vision Analysis Unavailable\n\n"
            "No vision model could process the screenshot. "
            "Here is how to enable it:\n\n"
            "### Option A — Ollama (local, free)\n"
            "```bash\n"
            "ollama pull llava\n"
            "# Then set in .env:\n"
            "VISION_MODEL=llava\n"
            "```\n\n"
            "### Option B — Groq (cloud, free tier)\n"
            "```\n"
            "# In .env:\n"
            "LLM_PROVIDER=groq\n"
            f"GROQ_VISION_MODEL={_GROQ_VISION_MODEL}\n"
            "GROQ_API_KEY=your_key_here\n"
            "```\n\n"
            f"Screenshot was saved to: `{path}`\n\n"
            f"Errors encountered: {'; '.join(errors)}"
        )
        return setup_guide

    elapsed = (time.perf_counter() - t0) * 1000
    log.info("[screen_analyzer] analysis complete in %.1f ms via %s", elapsed, backend_used)

    return (
        f"## Screen Analysis\n"
        f"**Screenshot**: `{path}` ({file_size_kb:.1f} KB)  \n"
        f"**Vision model**: {backend_used}  \n"
        f"**Analysis time**: {elapsed:.0f} ms\n\n"
        f"---\n\n"
        f"{analysis}"
    )
