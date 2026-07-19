"""
Conversation Memory — the "current chat only" memory type.

Simple in-process store keyed by session_id, mapping to a list of
LangChain messages. Swap this for a Redis/Postgres-backed store later
without touching the agent graph — it only calls get()/append()/clear().
"""

from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.logging_config import get_logger
from backend.config import settings

log = get_logger(__name__)


class ConversationMemory:
    def __init__(self):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._summaries: Dict[str, str] = {}

    def get(self, session_id: str) -> List[BaseMessage]:
        messages = self._sessions.setdefault(session_id, [])
        log.debug("[memory] get(session=%s) -> %d message(s)", session_id, len(messages))
        return messages

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def append(self, session_id: str, messages: List[BaseMessage]) -> None:
        session_messages = self._sessions.setdefault(session_id, [])
        session_messages.extend(messages)
        log.debug(
            "[memory] append(session=%s, +%d messages) -> %d total before pruning",
            session_id, len(messages), len(session_messages),
        )
        self._prune_and_summarize(session_id)

    def clear(self, session_id: str) -> None:
        had = len(self._sessions.get(session_id, []))
        self._sessions[session_id] = []
        log.debug("[memory] clear(session=%s) (dropped %d message(s))", session_id, had)

    def clear_summary(self, session_id: str) -> None:
        self._summaries[session_id] = ""
        log.debug("[memory] clear_summary(session=%s) (cleared conversation summary)", session_id)

    def _prune_and_summarize(self, session_id: str) -> None:
        messages = self._sessions.get(session_id, [])
        # Find indices of HumanMessage
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        
        max_turns = settings.MEMORY_MAX_TURNS
        if len(human_indices) > max_turns:
            # We want to keep the last `max_turns` human messages and all messages after them.
            # The split point will be the index of the human message that starts the window.
            prune_to_idx = human_indices[len(human_indices) - max_turns]
            messages_to_prune = messages[:prune_to_idx]
            active_messages = messages[prune_to_idx:]
            
            if messages_to_prune:
                old_summary = self._summaries.get(session_id, "")
                new_chunk_summary = self._generate_summary_of_messages(messages_to_prune)
                if new_chunk_summary:
                    if old_summary:
                        new_summary = old_summary + "\n" + new_chunk_summary
                    else:
                        new_summary = new_chunk_summary
                    
                    # Consolidate summary if it gets too long (e.g. over 400 words)
                    if len(new_summary.split()) > 400:
                        new_summary = self._consolidate_summary(new_summary)
                        
                    self._summaries[session_id] = new_summary
                    log.info(
                        "[memory] Session %s pruned %d message(s). Summary length is now %d chars.",
                        session_id, len(messages_to_prune), len(new_summary),
                    )
            
            self._sessions[session_id] = active_messages

    def _generate_summary_of_messages(self, messages: List[BaseMessage]) -> str:
        try:
            from backend.llm.factory import get_llm
            llm = get_llm()
            
            formatted_history = self._format_messages_for_summary(messages)
            
            prompt = (
                "Write a concise, bullet-point summary of the following conversation snippet. "
                "Focus on key facts discussed, user preferences, decisions made, or actions taken. "
                "Keep it under 3-4 bullet points and less than 80 words. Do not include intro or outro.\n\n"
                f"{formatted_history}"
            )
            
            response = llm.invoke([SystemMessage(content=prompt)])
            summary = response.content.strip()
            return summary
        except Exception:
            log.exception("[memory] Failed to generate conversation snippet summary")
            return ""

    def _consolidate_summary(self, summary: str) -> str:
        try:
            from backend.llm.factory import get_llm
            llm = get_llm()
            
            prompt = (
                "You are tasked with consolidating a long list of historical conversation summary points. "
                "Combine redundant points, preserve all critical facts, user identity, preferences, decisions, "
                "and current status. Keep it as a concise, structured bulleted list under 200 words. "
                "Do not include intro or outro.\n\n"
                f"Historical Summary:\n{summary}"
            )
            
            response = llm.invoke([SystemMessage(content=prompt)])
            consolidated = response.content.strip()
            log.info("[memory] Consolidated summary from %d to %d chars", len(summary), len(consolidated))
            return consolidated
        except Exception:
            log.exception("[memory] Failed to consolidate summary")
            return summary

    def _format_messages_for_summary(self, messages: List[BaseMessage]) -> str:
        formatted = []
        for msg in messages:
            role = msg.type
            content = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                calls = ", ".join(f"{c['name']}({c['args']})" for c in msg.tool_calls)
                content = f"[Calls tools: {calls}]"
            elif role == "tool":
                # Truncate tool response to prevent token bloat during summarization
                content_str = str(content)
                if len(content_str) > 200:
                    content = f"[Tool output truncated: {content_str[:200]}...]"
                else:
                    content = f"[Tool output: {content_str}]"
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)


# One shared instance for the whole app (simple process-local memory).
conversation_memory = ConversationMemory()
