"""
Tool Router.

The LLM itself decides if a tool is needed and which one, via native
tool-calling (bind_tools). This module just inspects the LLM's response
and routes the graph accordingly — it's a thin wrapper around LangGraph's
built-in `tools_condition` so the routing logic has a clear, named home
that matches the architecture doc (and is easy to customize later, e.g.
to add a hard-coded rule like "always use calculator for math questions").
"""

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from backend.tools.registry import ALL_TOOLS
from backend.logging_config import get_logger

log = get_logger(__name__)


_STATUS_HINTS = (
    "system status",
    "status",
    "snapshot",
    "current working directory",
    "current directory",
    "cwd",
    "environment",
)


def _last_user_text(state) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "")
    return ""


def _has_tool(name: str) -> bool:
    return any(tool.name == name for tool in ALL_TOOLS)


def route_after_agent(state):
    """Return 'tools' if the agent's last message requested a tool call,
    otherwise route to END so the graph responds directly."""
    decision = tools_condition(state)
    if decision == END:
        log.debug("[router] decision=END (no tool call requested)")
    else:
        log.debug("[router] decision=%s (routing to tool executor)", decision)
    return decision
