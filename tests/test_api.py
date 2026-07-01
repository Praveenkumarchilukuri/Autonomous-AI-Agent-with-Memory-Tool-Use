"""FastAPI endpoint tests — agent is mocked so no LLM calls made."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


MOCK_STATE = {
    "task": "What is 2+2?",
    "final_answer": "The answer is 4.",
    "trace": [
        {
            "iteration": 1,
            "thought": "I'll use Python to compute this.",
            "action": "python_repl",
            "action_input": "print(2+2)",
            "result": "4",
            "latency_ms": 120,
        },
        {
            "iteration": 2,
            "thought": "I have the answer.",
            "action": "finish",
            "action_input": "The answer is 4.",
            "result": "The answer is 4.",
            "latency_ms": 95,
        },
    ],
    "iteration": 2,
    "done": True,
    "memory_context": "No relevant memories.",
    "reasoning": "",
    "tool_name": "finish",
    "tool_input": "The answer is 4.",
    "tool_result": "The answer is 4.",
}


@pytest.fixture
def client():
    with patch("agent.graph.run_agent", return_value=MOCK_STATE), \
         patch("memory.store.MemoryStore") as MockStore:
        MockStore.return_value.size = 5
        MockStore.return_value.search.return_value = [
            {"text": "cached result", "metadata": {}, "score": 0.88}
        ]
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "tools" in d
        assert "model" in d


class TestAgentRun:
    def test_basic(self, client):
        r = client.post("/agent/run", json={"task": "What is 2+2?"})
        assert r.status_code == 200
        d = r.json()
        assert d["final_answer"] == "The answer is 4."
        assert d["success"] is True
        assert d["iterations"] == 2
        assert len(d["trace"]) == 2

    def test_trace_fields(self, client):
        r = client.post("/agent/run", json={"task": "What is 2+2?"})
        step = r.json()["trace"][0]
        assert "thought" in step
        assert "action" in step
        assert "result" in step
        assert "latency_ms" in step

    def test_empty_task_rejected(self, client):
        r = client.post("/agent/run", json={"task": ""})
        assert r.status_code == 422

    def test_task_too_long_rejected(self, client):
        r = client.post("/agent/run", json={"task": "x" * 5000})
        assert r.status_code == 422


class TestMemoryEndpoints:
    def test_search(self, client):
        r = client.post("/memory/search", json={"query": "France GDP", "k": 3})
        assert r.status_code == 200
        d = r.json()
        assert "results" in d
        assert "count" in d

    def test_clear(self, client):
        r = client.delete("/memory/clear")
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"

    def test_size(self, client):
        r = client.get("/memory/size")
        assert r.status_code == 200
        assert "size" in r.json()
