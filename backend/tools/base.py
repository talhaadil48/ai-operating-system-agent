"""
Every tool in this OS is just a LangChain @tool-decorated function.
This file only documents the convention so new tools stay consistent:

    from langchain_core.tools import tool

    @tool
    def my_tool(some_arg: str) -> str:
        '''One clear docstring — the LLM reads this to decide when to call it.'''
        ...
        return "result as a string"

Rules for adding a new tool:
  1. Put it in its own file in backend/tools/
  2. Give it a clear name + docstring (the LLM uses the docstring to decide
     whether/when to call it — treat it like an API description)
  3. Register it in backend/tools/registry.py -> ALL_TOOLS list
That's it. The agent graph picks up every tool in ALL_TOOLS automatically.
"""
