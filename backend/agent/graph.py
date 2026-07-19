"""
The Agent Graph — this IS the "single agent" from the architecture.

One LLM, bound to the tool registry, making every decision. LangGraph
just gives us a clean way to express the loop:

    START -> agent -> (needs tool?) -> tools -> agent -> ... -> END

Flow matches the architecture doc:
    Receive Input -> (memory already merged into messages by caller)
    -> Agent decides if tool needed -> Router -> Executor (tools node)
    -> back to Agent -> Final Response -> END
"""

from langgraph.graph import END, START, StateGraph

from backend.agent.executor import tool_executor_node
from backend.agent.router import route_after_agent
from backend.agent.state import AgentState
from backend.llm.factory import get_llm
from backend.logging_config import get_logger, log_timing
from backend.tools.registry import ALL_TOOLS

log = get_logger(__name__)
SYSTEM_PROMPT = (
    "You are a helpful, accurate AI assistant in a modular single-agent AI operating system.\n\n"
    "Tools:\n"
    "- calculator: Arithmetic.\n"
    "- system_status: OS, Python version, CWD, LLM config.\n"
    "- web_search: Search internet (news, docs, APIs, external info).\n"
    "- search_workspace: Search project files/code by text.\n"
    "- knowledge_base_search: Search uploaded documents (RAG).\n"
    "- run_shell_command: Run any shell/terminal command, capture stdout+stderr.\n"
    "- kill_process: Kill a process by PID.\n"
    "- read_file: Read a file's contents.\n"
    "- write_file: Write/create a file.\n"
    "- copy_file: Copy a file or directory.\n"
    "- move_file: Move or rename a file or directory.\n"
    "- delete_file: Delete a file or directory permanently.\n"
    "- list_directory: List directory contents.\n"
    "- search_files: Find files by name pattern or content.\n"
    "- list_processes: List running processes (filter by name).\n"
    "- get_system_resource_usage: Get CPU, RAM, and disk usage.\n"
    "- start_program: Launch a program/process in the background.\n"
    "- stop_process: Gracefully stop a process by PID.\n\n"
    "Rules:\n"
    "1. Only use tools when necessary. Answer directly if you know the answer.\n"
    "2. Use the single most relevant tool. Do not repeat tool calls or search for known info.\n"
    "3. Stop calling tools and answer immediately once you have enough information."
)
def _preview(text: str, n: int = 160) -> str:
    """Shorten a string for one-line log output."""
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def build_agent_node():
    """Bind tools to the LLM once, return a node function that calls it."""
    llm = get_llm()
    tool_names = [t.name for t in ALL_TOOLS] if ALL_TOOLS else []
    log.info("Agent node built. tools=%s", tool_names or "none")
    llm_with_tools = llm.bind_tools(ALL_TOOLS) if ALL_TOOLS else llm

    def agent_node(state: AgentState):
        messages = list(state["messages"])
        summary = state.get("summary", "")

        # Always ensure system prompt is present and up-to-date.
        from langchain_core.messages import SystemMessage
        system_content = SYSTEM_PROMPT
        if summary:
            system_content += f"\n\nHere is a summary of the conversation so far:\n{summary}"

        if messages and messages[0].type == "system":
            messages[0] = SystemMessage(content=system_content)
        else:
            messages = [SystemMessage(content=system_content)] + messages

        log.info("[agent] invoking LLM with %d message(s) in context", len(messages))
        log.debug(
            "[agent] full message stack:\n%s",
            "\n".join(f"  {i}. {m.type:<9} {_preview(getattr(m, 'content', ''), 300)}"
                      for i, m in enumerate(messages)),
        )

        try:
            with log_timing(log, "llm_invoke", level=10):  # DEBUG level timing
                response = llm_with_tools.invoke(messages)
        except Exception:
            log.exception("[agent] LLM invocation raised an exception")
            raise

        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            log.info(
                "[agent] LLM requested %d tool call(s): %s",
                len(tool_calls),
                ", ".join(f"{tc['name']}({tc.get('args')})" for tc in tool_calls),
            )
        else:
            log.info("[agent] LLM answered directly (no tool call). reply=%r",
                      _preview(response.content))

        return {"messages": [response]}

    return agent_node


def build_graph():
    log.info("Building agent graph (START -> agent -> tools? -> agent -> END)")
    graph = StateGraph(AgentState)

    graph.add_node("agent", build_agent_node())
    graph.add_node("tools", tool_executor_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    # After a tool runs, always go back to the agent so it can read the
    # tool result and produce the final answer (or call another tool).
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    log.info("Agent graph compiled and ready.")
    return compiled


# Compiled once and reused for every request.
agent_graph = build_graph()
