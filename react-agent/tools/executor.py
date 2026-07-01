"""Tool executor — string-input dispatcher used by agent nodes."""

from __future__ import annotations

import logging

from tools.web_search import web_search_tool
from tools.python_repl import python_repl_tool
from tools.wikipedia import wikipedia_tool

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Routes tool-name → function. Inputs and outputs are plain strings."""

    TOOL_MAP = {
        "web_search": web_search_tool,
        "python_repl": python_repl_tool,
        "wikipedia": wikipedia_tool,
        "finish": None,
    }

    def run(self, tool_name: str, tool_input: str) -> str:
        tool_name = (tool_name or "").strip().lower()

        if tool_name == "finish":
            return tool_input  # The input IS the final answer

        fn = self.TOOL_MAP.get(tool_name)
        if fn is None:
            available = ", ".join(k for k in self.TOOL_MAP if k != "finish")
            return f"Unknown tool '{tool_name}'. Available: {available}"

        logger.info(f"Tool: {tool_name} | Input: {str(tool_input)[:120]}")
        try:
            return fn(tool_input) or "(Tool returned empty output)"
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}", exc_info=True)
            return f"Tool error ({tool_name}): {e}"

    def list_tools(self) -> list[str]:
        return list(self.TOOL_MAP.keys())


_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
