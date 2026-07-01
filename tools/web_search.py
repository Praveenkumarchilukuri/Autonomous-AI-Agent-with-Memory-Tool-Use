"""
Web search tool.

Priority chain:
  1. Tavily (if TAVILY_API_KEY set and tavily-python installed)
  2. DuckDuckGo (if duckduckgo-search installed)
  3. Mock stub (always available — useful for tests / CI)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web and return formatted results.

    Args:
        query: Search query string.
        max_results: Number of results.

    Returns:
        Formatted results string.
    """
    if not query or not query.strip():
        return "Error: Empty search query."

    # ── Tavily ────────────────────────────────────────────────────────
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            resp = client.search(query.strip(), max_results=max_results, include_answer=True)
            parts = []
            if resp.get("answer"):
                parts.append(f"**Summary**: {resp['answer']}\n")
            for i, r in enumerate(resp.get("results", []), 1):
                parts.append(
                    f"{i}. **{r.get('title', 'No title')}**\n"
                    f"   {r.get('url', '')}\n"
                    f"   {r.get('content', '')[:400]}\n"
                )
            return "\n".join(parts) or "No results found."
        except Exception as e:
            logger.warning(f"Tavily failed ({e}); trying DuckDuckGo.")

    # ── DuckDuckGo ────────────────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query.strip(), max_results=max_results))
        if results:
            parts = [f"Search results for: '{query}'\n"]
            for i, r in enumerate(results, 1):
                parts.append(
                    f"{i}. **{r.get('title', 'No title')}**\n"
                    f"   {r.get('href', '')}\n"
                    f"   {r.get('body', '')[:400]}\n"
                )
            return "\n".join(parts)
    except Exception as e:
        logger.warning(f"DuckDuckGo failed ({e}); using mock.")

    # ── Mock stub ─────────────────────────────────────────────────────
    return (
        f"[MOCK] Web search for: '{query}'\n"
        "No live search backend configured.\n"
        "Set TAVILY_API_KEY or install duckduckgo-search for real results."
    )


class WebSearchTool:
    """Class-based wrapper for tool registry compatibility."""

    name = "web_search"
    description = (
        "Search the live web for current information. "
        "Input: {\"query\": str, \"max_results\": int (optional, default 5)}. "
        "Use for: recent events, statistics, facts, news."
    )

    def run(self, tool_input: dict) -> tuple[str, str | None]:
        query = tool_input.get("query", "")
        max_results = int(tool_input.get("max_results", 5))
        if not query:
            return "", "Missing 'query' parameter."
        result = web_search_tool(query, max_results)
        return result, None
