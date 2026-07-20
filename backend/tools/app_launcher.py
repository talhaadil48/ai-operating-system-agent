import os
import sys
import platform
import subprocess
from langchain_core.tools import tool
from backend.logging_config import get_logger

log = get_logger(__name__)

@tool
def open_app_or_url(target: str) -> str:
    """Open a desktop application, website URL, file, or folder.

    Use this when the user asks to open, run, start, or launch a program (e.g., 'notepad',
    'chrome', 'calculator'), open a folder, show a file, or visit a website (e.g., 'https://google.com').

    Args:
        target: The name of the application (e.g., 'notepad', 'calc'), path to file/folder, or URL.
    """
    log.info("[launcher] Attempting to open target: %r", target)
    try:
        # Check if it looks like a URL
        if target.startswith(("http://", "https://", "www.")):
            url = target if target.startswith("http") else "https://" + target
            if platform.system() == "Windows":
                os.startfile(url)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
            return f"Successfully opened URL: {url}"

        # Otherwise, try opening as application/file/folder
        if platform.system() == "Windows":
            # Check common app names and resolve executable if needed
            # e.g., if user inputs 'chrome' we can try chrome.exe, etc.
            app_lower = target.lower().strip()
            # Common mappings for ease of use
            app_map = {
                "chrome": "chrome.exe",
                "notepad": "notepad.exe",
                "calc": "calc.exe",
                "calculator": "calc.exe",
                "explorer": "explorer.exe",
                "paint": "mspaint.exe",
                "mspaint": "mspaint.exe",
                "cmd": "cmd.exe",
                "powershell": "powershell.exe"
            }
            exe_target = app_map.get(app_lower, target)
            os.startfile(exe_target)
            return f"Successfully opened '{exe_target}'"
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", "-a", target])
            return f"Successfully launched macOS app: '{target}'"
        else:  # Linux
            subprocess.Popen(["xdg-open", target])
            return f"Successfully launched target: '{target}'"
    except Exception as exc:
        log.exception("[launcher] Failed to open target: %s", target)
        return f"Error opening '{target}': {exc}. Check if the application name/path is correct."
