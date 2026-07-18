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
from langgraph.prebuilt import ToolNode

from backend.agent.executor import tool_executor_node
from backend.agent.router import route_after_agent
from backend.agent.state import AgentState
from backend.llm.factory import get_llm
from backend.logging_config import get_logger, log_timing
from backend.config import settings
from backend.rag.retrieve import retrieve_context
from backend.tools.registry import ALL_TOOLS

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, capable AI assistant that is part of a modular "
    "single-agent AI operating system. You have access to tools — use "
    "them whenever they help you answer accurately. For arithmetic, use "
    "calculator instead of computing it yourself. For current environment or "
    "status questions, use system_status. For questions about how this codebase works, "
    "files, structure, or implementation, always use the search_workspace tool to "
    "inspect the codebase first instead of guessing. For uploaded documents or indexed "
    "knowledge, use knowledge_base_search or rely on the provided retrieved "
    "context. If no tool is needed, just answer directly and concisely."
)


_STATUS_HINTS = (
    "system status",
    "status",
    "snapshot",
    "current working directory",
    "current directory",
    "cwd",
    "environment",
)

_RAG_HINTS = (
    "uploaded document",
    "uploaded docs",
    "uploaded file",
    "document",
    "documents",
    "knowledge base",
    "knowledgebase",
    "kb",
    "pdf",
    "manual",
    "policy",
    "notes",
)

_CODEBASE_HINTS = (
    "add tool",
    "new tool",
    "create tool",
    "register tool",
    "tools registry",
    "registry.py",
    "codebase",
    "file structure",
    "project structure",
)


def _preview(text: str, n: int = 160) -> str:
    """Shorten a string for one-line log output."""
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "…"


def _last_user_text(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "")
    return ""


def build_agent_node():
    """Bind tools to the LLM once, return a node function that calls it."""
    llm = get_llm()
    tool_names = [t.name for t in ALL_TOOLS] if ALL_TOOLS else []
    log.info("Agent node built. tools=%s", tool_names or "none")
    llm_with_tools = llm.bind_tools(ALL_TOOLS) if ALL_TOOLS else llm

    def agent_node(state: AgentState):
        messages = list(state["messages"])
        # Ensure the system prompt is always present as the first message, and always use the latest SYSTEM_PROMPT.
        if messages and messages[0].type == "system":
            from langchain_core.messages import SystemMessage
            messages[0] = SystemMessage(content=SYSTEM_PROMPT)
        else:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        user_text = _last_user_text(messages).lower()
        if any(hint in user_text for hint in _STATUS_HINTS):
            from langchain_core.messages import AIMessage
            from backend.tools.system_status import system_status

            log.info("[agent] answering status-style request directly via system_status")
            return {"messages": [AIMessage(content=system_status.func())]}

        if any(hint in user_text for hint in _CODEBASE_HINTS):
            from backend.tools.search_workspace import search_workspace
            search_results = search_workspace.func("registry")
            if search_results and "No matches found" not in search_results:
                from langchain_core.messages import SystemMessage
                context_message = SystemMessage(
                    content=(
                        "Actual codebase files and comments about tool registration:\n"
                        + search_results
                    )
                )
                # Inject right after system prompt
                messages = messages[:1] + [context_message] + messages[1:]
                log.info("[agent] injected codebase search context into prompt")

        if any(hint in user_text for hint in _RAG_HINTS):
            rag_context = retrieve_context(user_text, k=settings.RAG_TOP_K)
            if rag_context:
                from langchain_core.messages import SystemMessage

                context_message = SystemMessage(
                    content=(
                        "Relevant uploaded-document context:\n"
                        + "\n\n".join(rag_context)
                    )
                )
                messages = messages[:1] + [context_message] + messages[1:]
                log.info("[agent] injected %d RAG context chunk(s) into prompt", len(rag_context))

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
