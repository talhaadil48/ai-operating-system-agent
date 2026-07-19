"""
File Manager tool — lets the AI read, write, copy, move, delete, search,
and list files/directories in the local filesystem.

Security note: Only deploy in trusted environments or behind an auth layer.
"""

import os
import shutil
import fnmatch
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from backend.logging_config import get_logger

log = get_logger(__name__)

_MAX_READ_CHARS = 8000  # Truncate large files to prevent context bloat
_MAX_SEARCH_RESULTS = 30


@tool
def read_file(path: str) -> str:
    """Read the contents of a file and return it as text.

    Use this when the user wants to view a file's content or when you need to
    inspect code, configs, or documents.

    Args:
        path: Absolute or relative path to the file.
    """
    log.info("[files] Reading file: %s", path)
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_READ_CHARS:
            content = content[:_MAX_READ_CHARS] + f"\n\n... [truncated — file has {len(content)} chars total]"
        return content
    except FileNotFoundError:
        return f"File not found: {path}"
    except IsADirectoryError:
        return f"'{path}' is a directory. Use list_directory to browse it."
    except Exception as exc:
        log.exception("[files] Failed to read file: %s", path)
        return f"Error reading file: {exc}"


@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write text content to a file. Creates the file (and parent directories)
    if they do not exist.

    Use this when the user wants to create or update a file.

    Args:
        path: Absolute or relative path to write to.
        content: The text content to write.
        overwrite: If False (default), refuse to overwrite existing files.
    """
    log.info("[files] Writing file: %s (overwrite=%s)", path, overwrite)
    try:
        p = Path(path)
        if p.exists() and not overwrite:
            return (f"File already exists: {path}. "
                    f"Set overwrite=True to replace it.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written successfully: {path} ({len(content)} chars)"
    except Exception as exc:
        log.exception("[files] Failed to write file: %s", path)
        return f"Error writing file: {exc}"


@tool
def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory from source to destination.

    Args:
        source: Path to the file or folder to copy.
        destination: Destination path (file or directory).
    """
    log.info("[files] Copying %s -> %s", source, destination)
    try:
        src = Path(source)
        if not src.exists():
            return f"Source does not exist: {source}"
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        return f"Copied: {source} -> {destination}"
    except Exception as exc:
        log.exception("[files] Failed to copy: %s -> %s", source, destination)
        return f"Error copying: {exc}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory.

    Args:
        source: Path to the source file or directory.
        destination: Destination path.
    """
    log.info("[files] Moving %s -> %s", source, destination)
    try:
        if not Path(source).exists():
            return f"Source does not exist: {source}"
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
        return f"Moved: {source} -> {destination}"
    except Exception as exc:
        log.exception("[files] Failed to move: %s -> %s", source, destination)
        return f"Error moving: {exc}"


@tool
def delete_file(path: str) -> str:
    """Delete a file or an empty/non-empty directory permanently.

    Use with caution — this is irreversible.

    Args:
        path: Path to the file or directory to delete.
    """
    log.info("[files] Deleting: %s", path)
    try:
        p = Path(path)
        if not p.exists():
            return f"Nothing to delete — path does not exist: {path}"
        if p.is_dir():
            shutil.rmtree(path)
            return f"Directory deleted: {path}"
        else:
            p.unlink()
            return f"File deleted: {path}"
    except Exception as exc:
        log.exception("[files] Failed to delete: %s", path)
        return f"Error deleting: {exc}"


@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory (files and subdirectories).

    Use this when the user wants to browse a folder's contents or understand
    the project/directory structure.

    Args:
        path: Path to the directory to list (defaults to current directory).
    """
    log.info("[files] Listing directory: %s", path)
    try:
        p = Path(path)
        if not p.exists():
            return f"Directory not found: {path}"
        if not p.is_dir():
            return f"'{path}' is a file, not a directory. Use read_file to view it."

        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"Contents of: {p.resolve()}", ""]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  [DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                size_str = (
                    f"{size:,} B" if size < 1024
                    else f"{size / 1024:.1f} KB" if size < 1024 ** 2
                    else f"{size / (1024 ** 2):.1f} MB"
                )
                lines.append(f"  [FILE] {entry.name}  ({size_str})")
        lines.append(f"\nTotal: {len(entries)} item(s)")
        return "\n".join(lines)
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as exc:
        log.exception("[files] Failed to list directory: %s", path)
        return f"Error listing directory: {exc}"


@tool
def search_files(root_path: str, pattern: str, content_query: str = "") -> str:
    """Search for files by name pattern and/or by text content within files.

    Use this to find files matching a glob pattern (e.g., '*.py', '*.json')
    or to search file contents for a specific string.

    Args:
        root_path: Directory to search in (e.g., '.' or '/project').
        pattern: Glob pattern to match filenames (e.g., '*.py', 'config*').
        content_query: Optional string to search for inside matching files.
    """
    log.info("[files] Searching '%s' for pattern='%s' content='%s'",
             root_path, pattern, content_query)
    try:
        matches = []
        exclude_dirs = {".git", "__pycache__", "venv", ".ai_os", "node_modules"}
        root = Path(root_path)

        for file_path in root.rglob(pattern):
            if any(part in exclude_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue

            if content_query:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    if content_query.lower() in text.lower():
                        # find the first matching line
                        for i, line in enumerate(text.splitlines(), 1):
                            if content_query.lower() in line.lower():
                                matches.append(f"{file_path}:{i}: {line.strip()}")
                                break
                except Exception:
                    continue
            else:
                matches.append(str(file_path))

            if len(matches) >= _MAX_SEARCH_RESULTS:
                break

        if not matches:
            query_info = f" containing '{content_query}'" if content_query else ""
            return f"No files found matching '{pattern}'{query_info} in '{root_path}'"

        result = f"Found {len(matches)} match(es):\n" + "\n".join(matches)
        if len(matches) >= _MAX_SEARCH_RESULTS:
            result += f"\n... [showing first {_MAX_SEARCH_RESULTS} results]"
        return result
    except Exception as exc:
        log.exception("[files] Search failed: %s", root_path)
        return f"Error searching files: {exc}"
