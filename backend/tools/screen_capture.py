"""
Screen Capture Tool  (Computer Interaction Layer — Tool #19)

Captures a screenshot of the user's desktop and returns:
  - The screenshot saved as a timestamped PNG in .ai_os/screenshots/
  - The absolute file path to the saved image
  - The active (foreground) window title
  - Screen resolution(s)

No API keys required. Uses only local libraries (Pillow + pygetwindow on
Windows, pyautogui as cross-platform fallback).

Design notes
------------
• The file is saved to .ai_os/screenshots/ so it persists and can be read by
  analyze_screen() in the same session.
• The tool returns metadata as plain text so the LLM can describe what it
  sees when combined with screen_analysis (Tool #20).
• pygetwindow is Windows-only; on macOS/Linux we fall back gracefully.
"""

import os
import platform
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from backend.logging_config import get_logger

log = get_logger(__name__)

# ── screenshot output directory ──────────────────────────────────────────────
_SCREENSHOT_DIR = Path(".ai_os") / "screenshots"


def _ensure_dir() -> Path:
    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return _SCREENSHOT_DIR


def _timestamped_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _ensure_dir() / f"screenshot_{ts}.png"


def _get_active_window() -> str:
    """Return the title of the foreground window (Windows-only, degrades gracefully)."""
    system = platform.system()
    if system == "Windows":
        try:
            import pygetwindow as gw  # type: ignore
            win = gw.getActiveWindow()
            return win.title if win else "Unknown"
        except Exception as exc:
            log.debug("[screen_capture] pygetwindow unavailable: %s", exc)
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value or "Unknown"
            except Exception as exc2:
                log.debug("[screen_capture] ctypes fallback failed: %s", exc2)
                return "Unknown (pygetwindow not installed)"
    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first process whose frontmost is true'],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout.strip() or "Unknown"
        except Exception:
            return "Unknown"
    else:
        # Linux — xdotool if available
        try:
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout.strip() or "Unknown"
        except Exception:
            return "Unknown"


def _get_resolutions() -> str:
    """Return screen resolution string(s)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return f"{w}x{h}"
    except Exception:
        try:
            import pyautogui  # type: ignore
            size = pyautogui.size()
            return f"{size.width}x{size.height}"
        except Exception:
            return "Unknown"


@tool
def capture_screen(save_path: str = "") -> str:
    """Capture a screenshot of the entire desktop.

    Returns the file path of the saved screenshot, the active window title,
    and the screen resolution. Use this whenever the user says things like
    'what is on my screen', 'take a screenshot', 'show me what is open',
    or 'what does my desktop look like'.

    Args:
        save_path: Optional custom path to save the screenshot. If empty,
                   a timestamped file is auto-created in .ai_os/screenshots/.
    """
    log.info("[screen_capture] capturing screenshot …")
    t0 = time.perf_counter()

    # ── take the screenshot ──────────────────────────────────────────────────
    try:
        from PIL import ImageGrab  # type: ignore
        img = ImageGrab.grab()
    except ImportError:
        try:
            import pyautogui  # type: ignore
            img = pyautogui.screenshot()
        except ImportError:
            msg = (
                "Screenshot failed: neither Pillow (Pillow>=10) nor pyautogui is installed. "
                "Run: pip install Pillow pyautogui"
            )
            log.error("[screen_capture] %s", msg)
            return msg

    # ── save to disk ─────────────────────────────────────────────────────────
    out_path = Path(save_path) if save_path else _timestamped_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")

    elapsed = (time.perf_counter() - t0) * 1000
    width, height = img.size

    # ── gather metadata ──────────────────────────────────────────────────────
    active_window = _get_active_window()
    resolution = _get_resolutions()

    log.info(
        "[screen_capture] screenshot saved to %s (%dx%d) in %.1f ms",
        out_path, width, height, elapsed,
    )

    result = (
        f"Screenshot captured successfully.\n"
        f"  Saved to   : {out_path.resolve()}\n"
        f"  Dimensions : {width}x{height} px\n"
        f"  Resolution : {resolution}\n"
        f"  Active window: {active_window}\n"
        f"  Capture time : {elapsed:.1f} ms\n\n"
        f"You can now call analyze_screen with path='{out_path.resolve()}' "
        f"to get an AI description of what is on the screen."
    )
    return result
