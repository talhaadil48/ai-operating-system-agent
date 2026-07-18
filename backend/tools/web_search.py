import requests

from langchain_core.tools import tool

from backend.logging_config import get_logger
from backend.config import settings

log = get_logger(__name__)

URL = "https://google.serper.dev/search"
SERVER_API_KEY = settings.SERPER_API_KEY
print("SERPER_API_KEY:", SERVER_API_KEY)

@tool
def web_search(query: str) -> str:
    """
    Search Google for external information.

    Use ONLY when the user needs information from the internet, such as:
    - current events
    - news
    - latest versions
    - external facts
    - websites
    - documentation
    - APIs

    Do NOT use this tool for:
    - questions about available tools
    - questions about your capabilities
    - questions about what you can do
    - explaining this AI system

    For capability questions, answer directly from your available tools list.
    """
    headers = {
        "X-API-KEY": SERVER_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
        "num": 5,
    }

    try:
        log.debug("Searching: %s", query)

        response = requests.post(
            URL,
            json=payload,
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        answer_box = data.get("answerBox")
        if answer_box:
            if answer_box.get("answer"):
                results.append(
                    f"Answer: {answer_box['answer']}"
                )

        for result in data.get("organic", []):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            results.append(
                f"""Title: {title}
Snippet: {snippet}
URL: {link}
"""
            )

        if not results:
            return "No search results found."

        return "\n\n".join(results)

    except Exception as e:
        log.exception("Search failed")
        return f"Search failed: {e}"