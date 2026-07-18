"""
Tool Registry.

This is the ONLY place the agent graph looks to know what tools it has.
To add a new tool later: write it in its own file (see base.py for the
convention), import it here, and add it to ALL_TOOLS.
"""

from backend.logging_config import get_logger
from backend.tools.calculator import calculator
from backend.tools.knowledge_base import knowledge_base_search
from backend.tools.search_workspace import search_workspace
from backend.tools.system_status import system_status

log = get_logger(__name__)

ALL_TOOLS = [
    calculator,
    system_status,
    search_workspace,
    knowledge_base_search,
]

log.info("Tool registry loaded: %s", [t.name for t in ALL_TOOLS])
