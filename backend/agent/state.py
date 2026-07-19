"""
The single piece of state that flows through the LangGraph graph.

Right now it's just the message list (standard LangGraph pattern —
`add_messages` handles appending new messages / tool results correctly).
Add fields here later (e.g. `user_id`, `retrieved_context`) as the OS grows.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    summary: str
