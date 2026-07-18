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
    "You are a helpful, accurate AI assistant that is part of a modular "
    "single-agent AI operating system.\n\n"

    "Available tools:\n"
    "- calculator: evaluate arithmetic expressions.\n"
    "- system_status: retrieve OS, Python version, working directory, and LLM configuration.\n"
    "- search_workspace: search files in the user's project.\n"
    "- knowledge_base_search: search uploaded documents in the RAG knowledge base.\n"
    "- web_search: search the internet for current or external information.\n\n"

    "Rules:\n"
    "1. First decide whether a tool is actually needed. If you already know the answer from your instructions or conversation context, answer directly.\n"
    "2. Use only the single most relevant tool unless the user's request genuinely requires multiple tools.\n"
    "3. Never call a tool just to confirm information you already know.\n"
    "4. Never repeat the same tool call or search for the same information unless the user explicitly asks you to.\n"
    "5. For arithmetic, always use calculator.\n"
    "6. For questions about the user's code or project, use search_workspace.\n"
    "7. For questions about uploaded documents, use knowledge_base_search.\n"
    "8. For runtime or environment information, use system_status.\n"
    "9. For current events, news, documentation, APIs, websites, package versions, or any information that requires internet access, use web_search.\n"
    "10. After receiving sufficient information from a tool, answer the user immediately instead of calling additional tools.\n"
    "11. If no tool is needed, answer directly and concisely."
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
