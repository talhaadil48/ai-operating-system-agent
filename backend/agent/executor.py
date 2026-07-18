"""
Executor.

Runs whichever tool the router picked and returns the result back into
the message stream as a ToolMessage. Thin wrapper around LangGraph's
prebuilt ToolNode, built from the shared tool registry so adding a new
tool (backend/tools/registry.py) automatically makes it executable here
too — no changes needed in this file.
"""

from langgraph.prebuilt import ToolNode

from backend.logging_config import get_logger, log_timing
from backend.tools.registry import ALL_TOOLS

log = get_logger(__name__)

_raw_tool_node = ToolNode(ALL_TOOLS)


def _preview(value, n: int = 200) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def tool_executor_node(state):
    """Thin logging wrapper around LangGraph's ToolNode.

    Logs which tool(s) the agent asked for (name + args, taken from the
    last AI message's tool_calls), runs them via the real ToolNode, then
    logs each result (or error) along with total execution time — so a
    failing/slow tool is immediately visible in the logs.
    """
    last_message = state["messages"][-1]
    requested = getattr(last_message, "tool_calls", None) or []
    for call in requested:
        log.info("[tools] executing %s(%s) [id=%s]", call["name"], call.get("args"), call.get("id"))

    with log_timing(log, f"tool_node ({len(requested)} call(s))"):
        try:
            result = _raw_tool_node.invoke(state)
        except Exception:
            log.exception("[tools] tool execution raised an unhandled exception")
            raise

    for msg in result.get("messages", []):
        status = getattr(msg, "status", "success")
        level = log.error if status == "error" else log.info
        level(
            "[tools] result <- %s (status=%s): %s",
            getattr(msg, "name", "?"),
            status,
            _preview(getattr(msg, "content", "")),
        )

    return result
