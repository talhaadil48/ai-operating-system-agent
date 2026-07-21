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
from backend.tools.web_search import web_search
from backend.tools.terminal import run_shell_command, kill_process
from backend.tools.file_manager import (
    read_file,
    write_file,
    copy_file,
    move_file,
    delete_file,
    list_directory,
    search_files,
)
from backend.tools.process_manager import (
    list_processes,
    get_system_resource_usage,
    start_program,
    stop_process,
)
from backend.tools.app_launcher import open_app_or_url
from backend.tools.screen_analyzer import analyze_screen
from backend.tools.web_utilities import (
    scrape_webpage,
    summarize_webpage,
    check_website_status,
    get_weather,
)
from backend.tools.memory_tool import (
    save_long_term_memory,
    recall_long_term_memories,
    delete_long_term_memory,
)

log = get_logger(__name__)

ALL_TOOLS = [
    # Core reasoning / info
    calculator,
    system_status,
    web_search,

    # Long-Term Memory (PostgreSQL)
    save_long_term_memory,
    recall_long_term_memories,
    delete_long_term_memory,


    # Workspace & knowledge base
    search_workspace,
    knowledge_base_search,

    # Terminal / Shell
    run_shell_command,
    kill_process,

    # File Manager
    read_file,
    write_file,
    copy_file,
    move_file,
    delete_file,
    list_directory,
    search_files,

    # Process Manager
    list_processes,
    get_system_resource_usage,
    start_program,
    stop_process,

    # System Interactions
    open_app_or_url,
    analyze_screen,

    # Web Utilities
    scrape_webpage,
    summarize_webpage,
    check_website_status,
    get_weather,
]

log.info("Tool registry loaded: %s", [t.name for t in ALL_TOOLS])
