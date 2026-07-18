"""
Conversation Memory — the "current chat only" memory type.

Simple in-process store keyed by session_id, mapping to a list of
LangChain messages. Swap this for a Redis/Postgres-backed store later
without touching the agent graph — it only calls get()/append()/clear().
"""

from typing import Dict, List

from langchain_core.messages import BaseMessage

from backend.logging_config import get_logger

log = get_logger(__name__)


class ConversationMemory:
    def __init__(self):
        self._sessions: Dict[str, List[BaseMessage]] = {}

    def get(self, session_id: str) -> List[BaseMessage]:
        messages = self._sessions.setdefault(session_id, [])
        log.debug("[memory] get(session=%s) -> %d message(s)", session_id, len(messages))
        return messages

    def append(self, session_id: str, messages: List[BaseMessage]) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)
        log.debug(
            "[memory] append(session=%s, +%d messages) -> %d total",
            session_id, len(messages), len(self._sessions[session_id]),
        )

    def clear(self, session_id: str) -> None:
        had = len(self._sessions.get(session_id, []))
        self._sessions[session_id] = []
        log.debug("[memory] clear(session=%s) (dropped %d message(s))", session_id, had)


# One shared instance for the whole app (simple process-local memory).
conversation_memory = ConversationMemory()
