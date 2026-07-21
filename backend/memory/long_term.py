"""
Long-Term Memory — "things the AI should remember about the user across sessions"
(e.g. "User likes FastAPI", "User's OS is Windows", "Database is PostgreSQL").

Backed by PostgreSQL using SQLAlchemy. Includes debug logging when saving/recalling
memories and token-cap protection to prevent exploding prompt context sizes.
"""

from typing import List, Dict, Any, Optional
from backend.logging_config import get_logger
from backend.config import settings
from backend.database.connection import get_session_factory, init_db
from backend.database.models import LongTermMemoryRecord

log = get_logger(__name__)


class LongTermMemory:
    def __init__(self):
        self._fallback_facts: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = False

    def _ensure_init(self):
        if not self._initialized:
            try:
                init_db()
                self._initialized = True
            except Exception as exc:
                log.warning("[long_term_memory] DB init check failed: %s. Using in-memory fallback.", exc)

    def remember(self, user_id: str, fact: str, category: str = "preference") -> Dict[str, Any]:
        """Save a new fact into long-term memory in PostgreSQL database."""
        self._ensure_init()
        user_id = user_id or "default"
        category = category.strip().lower() if category else "preference"
        fact_clean = fact.strip()

        if not fact_clean:
            log.warning("[long_term_memory] Received empty fact, skipping save.")
            return {}

        log.info(
            "========================================================================\n"
            "[DEBUG LOG] [LTM] SAVING TO POSTGRESQL DATABASE:\n"
            "  User ID  : %s\n"
            "  Category : %s\n"
            "  Fact     : %s\n"
            "========================================================================",
            user_id, category, fact_clean
        )

        try:
            SessionLocal = get_session_factory()
            with SessionLocal() as db:
                # Check for duplicates or near-identical facts to prevent duplicate inflation
                existing = db.query(LongTermMemoryRecord).filter(
                    LongTermMemoryRecord.user_id == user_id,
                    LongTermMemoryRecord.fact == fact_clean
                ).first()

                if existing:
                    log.info("[long_term_memory] [DEBUG] Fact already exists in PostgreSQL database (ID=%s). Skipping duplicate.", existing.id)
                    return existing.to_dict()

                record = LongTermMemoryRecord(
                    user_id=user_id,
                    category=category,
                    fact=fact_clean
                )
                db.add(record)
                db.commit()
                db.refresh(record)

                result = record.to_dict()
                log.info(
                    "[long_term_memory] [DEBUG] SUCCESS: Saved memory record to PostgreSQL database! ID=%s, user_id=%s, fact=%r",
                    record.id, user_id, fact_clean
                )
                return result
        except Exception as exc:
            log.exception("[long_term_memory] Failed to save memory to PostgreSQL database. Falling back to in-memory store.")
            # In-memory fallback
            mem_list = self._fallback_facts.setdefault(user_id, [])
            mem_item = {"user_id": user_id, "category": category, "fact": fact_clean, "fallback": True}
            mem_list.append(mem_item)
            return mem_item

    def recall(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Recall saved long-term memory facts for a user, capped by limit."""
        self._ensure_init()
        user_id = user_id or "default"
        max_limit = limit or settings.LTM_MAX_FACTS

        log.debug("[long_term_memory] Recalling facts for user_id=%s (limit=%d)", user_id, max_limit)

        try:
            SessionLocal = get_session_factory()
            with SessionLocal() as db:
                records = (
                    db.query(LongTermMemoryRecord)
                    .filter(LongTermMemoryRecord.user_id == user_id)
                    .order_by(LongTermMemoryRecord.created_at.desc())
                    .limit(max_limit)
                    .all()
                )
                results = [r.to_dict() for r in records]
                log.info(
                    "[long_term_memory] [DEBUG] Recalled %d fact(s) from PostgreSQL database for user_id=%s",
                    len(results), user_id
                )
                return results
        except Exception:
            log.warning("[long_term_memory] Recalling from PostgreSQL failed. Using fallback in-memory store.")
            fallback_items = self._fallback_facts.get(user_id, [])
            return fallback_items[:max_limit]

    def delete_memory(self, identifier: str, user_id: str = "default") -> bool:
        """Delete a specific memory by UUID or matching fact keywords in PostgreSQL database."""
        self._ensure_init()
        identifier_clean = identifier.strip()
        log.info("[long_term_memory] [DEBUG] Attempting to delete memory (identifier=%r) from PostgreSQL database", identifier_clean)
        try:
            SessionLocal = get_session_factory()
            with SessionLocal() as db:
                # 1. Try exact UUID match
                record = db.query(LongTermMemoryRecord).filter(
                    LongTermMemoryRecord.user_id == user_id,
                    LongTermMemoryRecord.id == identifier_clean
                ).first()

                # 2. If not found, try text/keyword match on fact column
                if not record:
                    # Clean search terms e.g. "favourite_color_is_blue" -> "blue"
                    search_term = identifier_clean.replace("_", " ")
                    # Extract last word or key descriptor if multiple words
                    words = [w for w in search_term.split() if len(w) > 2]
                    query = db.query(LongTermMemoryRecord).filter(LongTermMemoryRecord.user_id == user_id)
                    for w in words:
                        query = query.filter(LongTermMemoryRecord.fact.ilike(f"%{w}%"))
                    record = query.first()

                if record:
                    deleted_id = record.id
                    deleted_fact = record.fact
                    db.delete(record)
                    db.commit()
                    log.info("[long_term_memory] [DEBUG] Successfully deleted memory ID=%s (fact=%r) from PostgreSQL", deleted_id, deleted_fact)
                    return True

                log.warning("[long_term_memory] [DEBUG] No memory record matched identifier %r in PostgreSQL", identifier_clean)
                return False
        except Exception:
            log.exception("[long_term_memory] Failed to delete memory identifier %r from PostgreSQL", identifier_clean)
            return False


    def format_for_prompt(self, user_id: str) -> str:
        """
        Formats long-term memory into a concise string for inclusion in the Agent's system prompt,
        ensuring it stays well within token limits.
        """
        memories = self.recall(user_id, limit=settings.LTM_MAX_FACTS)
        if not memories:
            return ""

        formatted_lines = []
        word_count = 0
        max_words = settings.LTM_MAX_SUMMARY_WORDS

        for item in memories:
            category = item.get("category", "preference").upper()
            fact = item.get("fact", "").strip()
            line = f"- [{category}] {fact}"
            words = len(line.split())
            if word_count + words > max_words:
                log.info("[long_term_memory] Reached prompt token limit threshold (%d words). Truncating further facts.", word_count)
                break
            formatted_lines.append(line)
            word_count += words

        return "\n".join(formatted_lines)


# Shared singleton instance
long_term_memory = LongTermMemory()
