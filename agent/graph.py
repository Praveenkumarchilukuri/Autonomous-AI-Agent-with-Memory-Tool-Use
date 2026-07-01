"""Build and compile the LangGraph ReAct agent graph."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_agent_graph():
    """
    Construct and compile the ReAct agent as a LangGraph StateGraph.

    Topology:
        START → load_memory → reason → act ──(done?)──► save_memory → END
                                  ▲        │
                                  └────────┘  (not done)
    """
    from langgraph.graph import StateGraph, END

    from agent.state import AgentState
    from agent.nodes import (
        memory_node,
        reason_node,
        act_node,
        save_memory_node,
        should_continue,
    )

    g = StateGraph(AgentState)

    g.add_node("load_memory",  memory_node)
    g.add_node("reason",       reason_node)
    g.add_node("act",          act_node)
    g.add_node("save_memory",  save_memory_node)

    g.set_entry_point("load_memory")
    g.add_edge("load_memory", "reason")
    g.add_edge("reason", "act")
    g.add_conditional_edges(
        "act",
        should_continue,
        {"reason": "reason", "save_memory": "save_memory"},
    )
    g.add_edge("save_memory", END)

    return g.compile()


# ── Singleton ─────────────────────────────────────────────────────────────────

_graph = None


def get_agent():
    """Return a compiled agent graph (singleton — built once per process)."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


# ── Convenience runner ────────────────────────────────────────────────────────

def run_agent(task: str) -> dict[str, Any]:
    """
    Run the agent on a task and return the final state dict.

    Example::

        result = run_agent("What is the 17th Fibonacci number?")
        print(result["final_answer"])
        print(result["trace"])
    """
    graph = get_agent()
    initial = _initial_state(task)
    return graph.invoke(initial)


def stream_agent(task: str):
    """
    Yield LangGraph state snapshots as the agent reasons step by step.

    Each yielded item is a dict of the updated node outputs.
    """
    graph = get_agent()
    initial = _initial_state(task)
    yield from graph.stream(initial)


def _initial_state(task: str) -> dict[str, Any]:
    return {
        "task": task,
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
