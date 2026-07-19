"""
Terminal / Shell tool — lets the AI run shell commands, capture stdout/stderr,
and kill running processes by PID.

Security note: This gives the agent OS-level command execution. Only deploy
in trusted environments or behind an authorization layer.
"""

import subprocess
import os
import signal
from typing import Optional
from langchain_core.tools import tool
from backend.logging_config import get_logger

log = get_logger(__name__)

# Max characters of output returned to the agent (prevent context bloat)
_MAX_OUTPUT_CHARS = 4000
# Timeout for shell commands (seconds)
_COMMAND_TIMEOUT = 30


@tool
def run_shell_command(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output (stdout + stderr combined).

    Use this to execute terminal commands such as running scripts, installing
    packages, checking git status, compiling code, or any other OS-level task.
    Returns combined stdout and stderr. Large outputs are truncated.

    Args:
        command: The shell command to run (e.g. 'ls -la', 'python --version').
        timeout: Seconds to wait before killing the command (default 30).
    """
    log.info("[shell] Running command: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(timeout, _COMMAND_TIMEOUT),
            cwd=os.getcwd(),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output_parts = []
        if stdout:
            output_parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")
        if not output_parts:
            output_parts.append("(no output)")

        combined = "\n\n".join(output_parts)
        exit_code_line = f"Exit code: {result.returncode}"

        full_output = f"{combined}\n\n{exit_code_line}"

        if len(full_output) > _MAX_OUTPUT_CHARS:
            full_output = full_output[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        log.info("[shell] Command exit code: %d", result.returncode)
        return full_output

    except subprocess.TimeoutExpired:
        msg = f"Command timed out after {timeout}s: {command}"
        log.warning("[shell] %s", msg)
        return msg
    except Exception as exc:
        log.exception("[shell] Failed to run command: %s", command)
        return f"Error running command: {exc}"


@tool
def kill_process(pid: int) -> str:
    """Kill a running process by its PID (Process ID).

    Use this when you need to stop a specific process. The PID can be
    obtained from list_processes or run_shell_command('tasklist' on Windows,
    'ps aux' on Linux/Mac).

    Args:
        pid: The integer Process ID of the process to kill.
    """
    log.info("[shell] Killing PID %d", pid)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True, capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        return f"Process {pid} killed successfully."
    except ProcessLookupError:
        return f"No process with PID {pid} found."
    except PermissionError:
        return f"Permission denied: cannot kill PID {pid}."
    except Exception as exc:
        log.exception("[shell] Failed to kill PID %d", pid)
        return f"Error killing process {pid}: {exc}"
