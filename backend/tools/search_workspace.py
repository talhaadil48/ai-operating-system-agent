"""Workspace search tool."""

import os
from langchain_core.tools import tool

@tool
def search_workspace(query: str) -> str:
    """Search for a text string in the files within the workspace/codebase directory.
    Use this tool whenever the user asks questions about how this codebase works, how to
    implement or modify files, or needs to locate code strings/modules in the project files."""
    matches = []
    exclude_dirs = {".git", "venv", ".ai_os", "__pycache__", ".gemini"}
    exclude_exts = {".pyc", ".png", ".jpg", ".jpeg", ".zip", ".pdf", ".docx", ".exe", ".dll", ".so"}
    
    query_lower = query.lower()
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in exclude_exts:
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if query_lower in line.lower():
                            matches.append(f"{path}:{i}: {line.strip()}")
                            if len(matches) >= 20:
                                break
            except Exception:
                continue
            if len(matches) >= 20:
                break
        if len(matches) >= 20:
            break
            
    if not matches:
        return f"No matches found for '{query}' in workspace."
    return "\n".join(matches)
