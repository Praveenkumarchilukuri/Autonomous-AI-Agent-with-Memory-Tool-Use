"""Unit tests for agent nodes — LLM is mocked, no API key needed."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.state import AgentState


def _state(**kwargs) -> AgentState:
    base: AgentState = {
        "task": "Test task",
        "memory_context": "",
        "reasoning": "",
        "tool_name": "",
        "tool_input": "",
        "tool_result": "",
        "final_answer": "",
        "done": False,
        "iteration": 0,
        "trace": [],
    }
    base.update(kwargs)
    return base


# ── memory_node ───────────────────────────────────────────────────────────────

class TestMemoryNode:
    def test_returns_memory_context_key(self):
        from agent.nodes import memory_node
        with patch("memory.store.MemoryStore") as MockStore:
            MockStore.return_value.search.return_value = [
                {"text": "Past answer", "score": 0.9, "metadata": {}}
            ]
            result = memory_node(_state())
        assert "memory_context" in result
        assert "Past answer" in result["memory_context"]

    def test_empty_store_gives_no_memories(self):
        from agent.nodes import memory_node
        with patch("memory.store.MemoryStore") as MockStore:
            MockStore.return_value.search.return_value = []
            result = memory_node(_state())
        assert "No relevant memories" in result["memory_context"]

    def test_handles_store_exception(self):
        from agent.nodes import memory_node
        with patch("memory.store.MemoryStore", side_effect=Exception("db error")):
            result = memory_node(_state())
        assert "memory_context" in result


# ── reason_node ───────────────────────────────────────────────────────────────

class TestReasonNode:
    def _mock_llm(self, content: str):
        m = MagicMock()
        m.invoke.return_value = MagicMock(content=content)
        return m

    def test_chooses_tool(self):
        from agent.nodes import reason_node
        payload = json.dumps({
            "thought": "I need to search",
            "action": "web_search",
            "action_input": "LangGraph tutorial"
        })
        with patch("agent.nodes._get_llm", return_value=self._mock_llm(payload)):
            result = reason_node(_state())
        assert result["tool_name"] == "web_search"
        assert "LangGraph" in result["tool_input"]
        assert result["iteration"] == 1

    def test_finish_action(self):
        from agent.nodes import reason_node
        payload = json.dumps({
            "thought": "I have the answer",
            "action": "finish",
            "action_input": "The answer is 42."
        })
        with patch("agent.nodes._get_llm", return_value=self._mock_llm(payload)):
            result = reason_node(_state())
        assert result["tool_name"] == "finish"

    def test_malformed_json_graceful(self):
        from agent.nodes import reason_node
        with patch("agent.nodes._get_llm", return_value=self._mock_llm("not json at all")):
            result = reason_node(_state())
        assert "tool_name" in result   # should default to finish

    def test_trace_entry_added(self):
        from agent.nodes import reason_node
        payload = json.dumps({"thought": "t", "action": "finish", "action_input": "ans"})
        with patch("agent.nodes._get_llm", return_value=self._mock_llm(payload)):
            result = reason_node(_state())
        assert len(result["trace"]) == 1
        assert result["trace"][0]["iteration"] == 1


# ── act_node ──────────────────────────────────────────────────────────────────

class TestActNode:
    def test_runs_python_repl(self):
        from agent.nodes import act_node
        state = _state(tool_name="python_repl", tool_input="print(99)", iteration=1)
        result = act_node(state)
        assert "99" in result["tool_result"]
        assert result["done"] is False

    def test_finish_sets_done(self):
        from agent.nodes import act_node
        state = _state(tool_name="finish", tool_input="Final answer here", iteration=1)
        result = act_node(state)
        assert result["done"] is True
        assert result["final_answer"] == "Final answer here"

    def test_unknown_tool_handled(self):
        from agent.nodes import act_node
        state = _state(tool_name="nonexistent_tool", tool_input="x", iteration=1)
        result = act_node(state)
        assert "Unknown tool" in result["tool_result"]


# ── should_continue ───────────────────────────────────────────────────────────

class TestShouldContinue:
    def test_continues_when_not_done(self):
        from agent.nodes import should_continue
        result = should_continue(_state(done=False, iteration=3))
        assert result == "reason"

    def test_stops_when_done(self):
        from agent.nodes import should_continue
        result = should_continue(_state(done=True, iteration=3))
        assert result == "save_memory"

    def test_stops_at_max_iterations(self):
        from agent.nodes import should_continue
        from config import settings
        result = should_continue(_state(done=False, iteration=settings.max_iterations))
        assert result == "save_memory"


# ── save_memory_node ──────────────────────────────────────────────────────────

class TestSaveMemoryNode:
    def test_saves_when_answer_present(self):
        from agent.nodes import save_memory_node
        state = _state(task="Test task", final_answer="The answer is 42.", iteration=3)
        with patch("memory.store.MemoryStore") as MockStore:
            save_memory_node(state)
            MockStore.return_value.add.assert_called_once()

    def test_skips_when_no_answer(self):
        from agent.nodes import save_memory_node
        state = _state(task="Test task", final_answer="")
        with patch("memory.store.MemoryStore") as MockStore:
            save_memory_node(state)
            MockStore.return_value.add.assert_not_called()
