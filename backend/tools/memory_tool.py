"""
Tool for managing Long-Term Memory (LTM) in PostgreSQL database.
"""

from typing import Literal
from langchain_core.tools import tool
from backend.logging_config import get_logger
from backend.memory.long_term import long_term_memory

log = get_logger(__name__)


@tool
def save_long_term_memory(
    fact: str,
    category: str = "preference",
    user_id: str = "default"
) -> str:
    """Save a user preference, environment detail, rule, or persistent fact to PostgreSQL long-term memory.

    Args:
        fact: The specific statement or preference to remember (e.g. 'Prefers TypeScript and FastAPI', 'Local Postgres port is 5432').
        category: The category of the fact ('preference', 'environment', 'biography', 'rule'). Defaults to 'preference'.
        user_id: The ID of the user. Defaults to 'default'.
    """
    log.info("[tool:save_long_term_memory] Saving fact: category=%s, fact=%r", category, fact)
    res = long_term_memory.remember(user_id=user_id, fact=fact, category=category)
    if res:
        return f"Successfully saved to PostgreSQL long-term memory (ID={res.get('id', 'N/A')}): [{category}] {fact}"
    return "Failed to save to PostgreSQL long-term memory."


@tool
def recall_long_term_memories(user_id: str = "default") -> str:
    """Retrieve all long-term memories and preferences currently saved in PostgreSQL for a user.

    Args:
        user_id: The user ID to query. Defaults to 'default'.
    """
    log.info("[tool:recall_long_term_memories] Recalling facts for user_id=%s", user_id)
    memories = long_term_memory.recall(user_id=user_id)
    if not memories:
        return f"No long-term memories found in database for user '{user_id}'."

    formatted = [f"- [{m.get('category', 'pref').upper()}] (ID: {m.get('id')}) {m.get('fact')}" for m in memories]
    return f"Long-term memories for user '{user_id}':\n" + "\n".join(formatted)


@tool
def delete_long_term_memory(memory_id: str) -> str:
    """Delete a specific long-term memory record from PostgreSQL database by its unique memory_id.

    Args:
        memory_id: The UUID of the memory record to remove.
    """
    log.info("[tool:delete_long_term_memory] Deleting memory ID=%s", memory_id)
    success = long_term_memory.delete_memory(memory_id)
    if success:
        return f"Successfully deleted memory record ID={memory_id} from PostgreSQL database."
    return f"Failed to delete memory record ID={memory_id} (record not found or database error)."
