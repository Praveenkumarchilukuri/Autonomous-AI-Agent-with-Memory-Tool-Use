"""Sandboxed Python REPL — captures stdout, allows common stdlib."""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout, redirect_stderr


def python_repl_tool(code: str) -> str:
    """
    Execute Python code and return stdout.

    Args:
        code: Valid Python code string.

    Returns:
        Captured stdout or error traceback.
    """
    if not code or not code.strip():
        return "Error: No code provided."

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Safe globals with useful modules pre-imported
    import math, json, re, itertools, collections, statistics, datetime, functools
    safe_globals: dict = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "math": math,
        "json": json,
        "re": re,
        "itertools": itertools,
        "collections": collections,
        "statistics": statistics,
        "datetime": datetime,
        "functools": functools,
    }

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<repl>", "exec"), safe_globals)  # noqa: S102

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        result = out + (f"\nSTDERR:\n{err}" if err else "")
        return result.strip() or "(Executed with no output)"

    except Exception:
        return f"Execution Error:\n{traceback.format_exc(limit=6)}"


class PythonREPLTool:
    """Class-based wrapper for tool registry compatibility."""

    name = "python_repl"
    description = (
        "Execute Python code in a sandboxed environment. "
        "Input: {\"code\": str}. "
        "Available modules: math, json, re, itertools, collections, statistics, datetime. "
        "Use for: calculations, algorithms, data processing, number crunching."
    )

    def run(self, tool_input: dict) -> tuple[str, str | None]:
        code = tool_input.get("code", "").strip()
        if not code:
            return "", "Missing 'code' parameter."
        result = python_repl_tool(code)
        if result.startswith("Execution Error:"):
            return result, result
        return result, None
