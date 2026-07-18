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
    "You are a helpful, capable AI assistant that is part of a modular "
    "single-agent AI operating system. You have access to the following tools:\n"
    "- calculator: evaluate arithmetic expressions\n"
    "- system_status: get current OS, Python version, working directory, and LLM config\n"
    "- search_workspace: search all project files for any string or keyword\n"
    "- knowledge_base_search: search uploaded documents indexed in the RAG store\n\n"
    "Rules:\n"
    "1. Always use a tool when it will give a more accurate answer than guessing.\n"
    "2. For math, always use calculator — never compute it yourself.\n"
    "3. For questions about this codebase (files, structure, how things work), "
    "always call search_workspace to look it up — never guess or fabricate file paths.\n"
    "4. For questions about uploaded documents, use knowledge_base_search.\n"
    "5. For system/environment info, use system_status.\n"
    "6. If no tool is needed, answer directly and concisely."
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

        # Always ensure system prompt is present and up-to-date.
        from langchain_core.messages import SystemMessage
        if messages and messages[0].type == "system":
            messages[0] = SystemMessage(content=SYSTEM_PROMPT)
        else:
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

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
