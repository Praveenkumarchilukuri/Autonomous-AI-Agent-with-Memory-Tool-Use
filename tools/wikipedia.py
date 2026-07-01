"""Wikipedia lookup tool."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def wikipedia_tool(query: str, sentences: int = 8) -> str:
    """
    Fetch a Wikipedia article summary.

    Args:
        query: Article title or search term.
        sentences: Number of summary sentences.

    Returns:
        Formatted summary or error message.
    """
    if not query or not query.strip():
        return "Error: Empty Wikipedia query."

    try:
        import wikipedia as wiki
        wiki.set_lang("en")

        try:
            page = wiki.page(query.strip(), auto_suggest=True)
            summary = wiki.summary(query.strip(), sentences=sentences)
            return f"**{page.title}**\nURL: {page.url}\n\n{summary}"
        except wiki.DisambiguationError as e:
            first = e.options[0]
            summary = wiki.summary(first, sentences=sentences)
            page = wiki.page(first)
            return f"**{page.title}** (disambiguation → {first})\nURL: {page.url}\n\n{summary}"
        except wiki.PageError:
            results = wiki.search(query, results=3)
            if not results:
                return f"No Wikipedia article found for: {query}"
            best = results[0]
            summary = wiki.summary(best, sentences=sentences)
            return f"**{best}** (best match)\n\n{summary}"

    except ImportError:
        return (
            f"[MOCK] Wikipedia lookup: '{query}'\n"
            "Install 'wikipedia' package for real results: pip install wikipedia"
        )
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return f"Wikipedia error: {e}"


class WikipediaTool:
    """Class-based wrapper for tool registry compatibility."""

    name = "wikipedia"
    description = (
        "Look up encyclopedic information from Wikipedia. "
        "Input: {\"query\": str, \"sentences\": int (optional)}. "
        "Use for: definitions, history, people, concepts, science."
    )

    def run(self, tool_input: dict) -> tuple[str, str | None]:
        query = tool_input.get("query", tool_input.get("title", ""))
        sentences = int(tool_input.get("sentences", 8))
        if not query:
            return "", "Missing 'query' parameter."
        result = wikipedia_tool(query, sentences)
        return result, None
