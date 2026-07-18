"""
Planner — kept deliberately thin, as described in the architecture doc:
"it doesn't have to be another model, it's just your workflow logic."

Right now the single agent + tool-calling loop in graph.py handles simple
"call a tool, read result, answer" flows on its own, so there's nothing
extra to plan. This module is a placeholder for when you want explicit
multi-step task breakdown (e.g. "read PDF -> search web -> compare -> summarize")
instead of letting the LLM decide step-by-step.

Wire it into graph.py as a node once you need it — the graph doesn't
care how a step was decided, only that it returns messages/instructions.
"""


def plan(task: str) -> list[str]:
    """Return an ordered list of steps for a task. Currently a passthrough —
    the single step IS the task, and the agent+tools loop figures out the rest."""
    return [task]
