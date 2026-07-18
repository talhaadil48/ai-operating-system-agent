"""
Calculator tool.

Chosen as the "one tool that just works" because it needs no API keys,
no internet access, and no extra services — good for verifying the whole
agent -> router -> tool -> agent loop end to end before you add real tools
(web search, PDF reader, etc).

Uses Python's `ast` module to safely evaluate arithmetic expressions
instead of raw eval() (no arbitrary code execution).
"""

import ast
import operator

from langchain_core.tools import tool

from backend.logging_config import get_logger

log = get_logger(__name__)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) / 2'.
    Supports + - * / // % ** and parentheses. Use this whenever the user
    asks for a calculation instead of doing math yourself."""
    log.debug("[calculator] evaluating expression=%r", expression)
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        log.debug("[calculator] %r = %r", expression, result)
        return str(result)
    except Exception as exc:
        log.warning("[calculator] failed to evaluate %r: %s", expression, exc)
        return f"Error evaluating '{expression}': {exc}"
