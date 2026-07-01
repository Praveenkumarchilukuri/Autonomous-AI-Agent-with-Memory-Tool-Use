"""Unit tests for all tools — no API keys required."""

import pytest
from tools.python_repl import python_repl_tool, PythonREPLTool
from tools.web_search import WebSearchTool
from tools.wikipedia import WikipediaTool
from tools.executor import ToolExecutor


class TestPythonREPL:
    def test_basic_arithmetic(self):
        out = python_repl_tool("print(2 + 2)")
        assert "4" in out

    def test_math_factorial(self):
        out = python_repl_tool("import math\nprint(math.factorial(6))")
        assert "720" in out

    def test_loop_output(self):
        out = python_repl_tool("for i in range(3): print(i)")
        assert "0" in out and "2" in out

    def test_string_reverse(self):
        out = python_repl_tool("s='hello'\nprint(s[::-1])")
        assert "olleh" in out

    def test_fibonacci(self):
        code = (
            "a,b=0,1\n"
            "for _ in range(19): a,b=b,a+b\n"
            "print(b)"
        )
        out = python_repl_tool(code)
        assert "6765" in out

    def test_syntax_error(self):
        out = python_repl_tool("def broken(")
        assert "Error" in out or "error" in out

    def test_empty_code(self):
        out = python_repl_tool("")
        assert "Error" in out

    def test_class_tool_wrapper(self):
        t = PythonREPLTool()
        result, err = t.run({"code": "print('hi')"})
        assert "hi" in result
        assert err is None

    def test_class_tool_missing_code(self):
        t = PythonREPLTool()
        _, err = t.run({})
        assert err is not None


class TestWebSearch:
    def test_returns_string(self):
        t = WebSearchTool()
        out, err = t.run({"query": "LangGraph python"})
        assert isinstance(out, str)
        assert len(out) > 0
        assert err is None

    def test_missing_query(self):
        t = WebSearchTool()
        _, err = t.run({})
        assert err is not None


class TestWikipedia:
    def test_returns_string(self):
        t = WikipediaTool()
        out, err = t.run({"query": "Python programming language"})
        assert isinstance(out, str)
        assert len(out) > 0

    def test_missing_query(self):
        t = WikipediaTool()
        _, err = t.run({})
        assert err is not None


class TestToolExecutor:
    def setup_method(self):
        self.ex = ToolExecutor()

    def test_python_repl(self):
        out = self.ex.run("python_repl", "print(1+1)")
        assert "2" in out

    def test_finish_passthrough(self):
        out = self.ex.run("finish", "The answer is 42.")
        assert out == "The answer is 42."

    def test_unknown_tool(self):
        out = self.ex.run("nonexistent", "input")
        assert "Unknown tool" in out

    def test_list_tools(self):
        tools = self.ex.list_tools()
        assert "web_search" in tools
        assert "python_repl" in tools
        assert "wikipedia" in tools
        assert "finish" in tools
