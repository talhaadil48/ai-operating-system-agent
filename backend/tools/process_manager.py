"""
Process Manager tool — lets the AI list running processes, check CPU/RAM usage,
start programs, and stop processes by PID or name.

Requires: psutil (pip install psutil)
"""

import os
import subprocess
from typing import Optional
from langchain_core.tools import tool
from backend.logging_config import get_logger

log = get_logger(__name__)

_MAX_PROCESS_LIST = 40  # Max processes returned in listing


@tool
def list_processes(filter_name: str = "") -> str:
    """List all currently running processes, optionally filtered by name.

    Returns PID, name, CPU%, RAM usage, and status for each process.
    Use this when the user asks about running programs, resource usage, or
    wants to find the PID of a specific application.

    Args:
        filter_name: Optional process name to filter by (case-insensitive).
    """
    log.info("[procs] Listing processes (filter=%r)", filter_name)
    try:
        import psutil
        procs = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
            try:
                info = proc.info
                name = info.get("name") or ""
                if filter_name and filter_name.lower() not in name.lower():
                    continue
                mem_mb = (info.get("memory_info").rss / (1024 ** 2)
                          if info.get("memory_info") else 0)
                procs.append({
                    "pid": info["pid"],
                    "name": name,
                    "cpu": info.get("cpu_percent", 0.0),
                    "mem_mb": round(mem_mb, 1),
                    "status": info.get("status", "?"),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not procs:
            return "No matching processes found." if filter_name else "No processes found."

        # Sort by memory (most expensive first)
        procs.sort(key=lambda p: p["mem_mb"], reverse=True)
        procs = procs[:_MAX_PROCESS_LIST]

        lines = [f"{'PID':>7}  {'Name':<30}  {'CPU%':>5}  {'RAM(MB)':>8}  Status"]
        lines.append("-" * 65)
        for p in procs:
            lines.append(
                f"{p['pid']:>7}  {p['name']:<30}  {p['cpu']:>5.1f}  {p['mem_mb']:>8.1f}  {p['status']}"
            )
        if len(procs) >= _MAX_PROCESS_LIST:
            lines.append(f"... [showing top {_MAX_PROCESS_LIST} by RAM]")
        return "\n".join(lines)
    except ImportError:
        return "psutil is not installed. Run: pip install psutil"
    except Exception as exc:
        log.exception("[procs] Failed to list processes")
        return f"Error listing processes: {exc}"


@tool
def get_system_resource_usage() -> str:
    """Get overall system CPU and RAM usage, plus disk usage statistics.

    Use this when the user wants a quick health check of the system,
    asks about memory usage, or how much disk space is left.
    """
    log.info("[procs] Getting system resource usage")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)

        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)
        ram_used = ram.used / (1024 ** 3)
        ram_avail = ram.available / (1024 ** 3)
        ram_percent = ram.percent

        disk = psutil.disk_usage(os.getcwd())
        disk_total = disk.total / (1024 ** 3)
        disk_used = disk.used / (1024 ** 3)
        disk_free = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        return (
            f"=== System Resource Usage ===\n"
            f"CPU:  {cpu_percent:.1f}% used  ({cpu_count} logical cores)\n\n"
            f"RAM:  {ram_used:.2f} GB used / {ram_total:.2f} GB total  "
            f"({ram_percent:.1f}%)  —  {ram_avail:.2f} GB available\n\n"
            f"Disk: {disk_used:.2f} GB used / {disk_total:.2f} GB total  "
            f"({disk_percent:.1f}%)  —  {disk_free:.2f} GB free"
        )
    except ImportError:
        return "psutil is not installed. Run: pip install psutil"
    except Exception as exc:
        log.exception("[procs] Failed to get resource usage")
        return f"Error getting resource usage: {exc}"


@tool
def start_program(command: str) -> str:
    """Start a program or process in the background (non-blocking).

    Use this to launch applications, start servers, or run scripts without
    waiting for them to finish.

    Args:
        command: The program or command to start (e.g., 'notepad.exe',
                 'python server.py', 'npm start').
    """
    log.info("[procs] Starting program: %s", command)
    try:
        if os.name == "nt":
            proc = subprocess.Popen(
                command, shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            proc = subprocess.Popen(
                command, shell=True,
                start_new_session=True,
            )
        return f"Started '{command}' with PID {proc.pid}."
    except Exception as exc:
        log.exception("[procs] Failed to start program: %s", command)
        return f"Error starting program: {exc}"


@tool
def stop_process(pid: int) -> str:
    """Gracefully stop a running process by PID. Falls back to force-kill
    if the process does not terminate within 5 seconds.

    Use this to stop background servers, scripts, or any running process.

    Args:
        pid: The Process ID of the process to stop.
    """
    log.info("[procs] Stopping PID %d", pid)
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
            return f"Process {pid} ({proc.name()}) terminated gracefully."
        except psutil.TimeoutExpired:
            proc.kill()
            return f"Process {pid} force-killed (did not respond to termination)."
    except psutil.NoSuchProcess:
        return f"No process found with PID {pid}."
    except psutil.AccessDenied:
        return f"Permission denied: cannot stop PID {pid}."
    except ImportError:
        return "psutil is not installed. Run: pip install psutil"
    except Exception as exc:
        log.exception("[procs] Failed to stop PID %d", pid)
        return f"Error stopping process: {exc}"
