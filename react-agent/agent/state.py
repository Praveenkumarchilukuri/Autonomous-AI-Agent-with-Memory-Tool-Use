"""Agent state — pure TypedDict, no external dependencies."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """Mutable state threaded through every LangGraph node."""

    # ── Input ─────────────────────────────────────────────────────────
    task: str

    # ── Memory ────────────────────────────────────────────────────────
    memory_context: str          # retrieved from FAISS before first step

    # ── Current reasoning step ────────────────────────────────────────
    reasoning: str               # latest thought from LLM
    tool_name: str               # chosen tool
    tool_input: str              # tool argument (plain string)
    tool_result: str             # observation from tool

    # ── Final output ──────────────────────────────────────────────────
    final_answer: str
    done: bool

    # ── Loop counter ──────────────────────────────────────────────────
    iteration: int

    # ── Streaming trace (list is appended to each step) ───────────────
    trace: Annotated[list[dict[str, Any]], operator.add]
