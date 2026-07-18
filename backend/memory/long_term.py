"""
Long-Term Memory — "things the AI should remember about the user"
(e.g. "User likes Python", "Works at XYZ").

Stub for now. When you're ready, back this with a small DB or a vector
store and wire it into agent/graph.py the same way conversation memory
is wired in — the graph doesn't care how it's implemented.
"""


class LongTermMemory:
    def __init__(self):
        self._facts = {}

    def remember(self, user_id: str, fact: str) -> None:
        self._facts.setdefault(user_id, []).append(fact)

    def recall(self, user_id: str) -> list:
        return self._facts.get(user_id, [])


long_term_memory = LongTermMemory()
