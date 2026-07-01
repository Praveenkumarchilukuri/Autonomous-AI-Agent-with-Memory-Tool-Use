"""LangGraph node functions: memory → reason → act → save_memory."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from agent.prompts import SYSTEM_PROMPT, TASK_TEMPLATE
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from config import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
        )
    return _llm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_history(trace: list[dict]) -> str:
    if not trace:
        return "None yet."
    lines = []
    for step in trace:
        n = step.get("iteration", "?")
        thought = step.get("thought", "")
        action = step.get("action", "")
        action_input = step.get("action_input", "")
        result = step.get("result", "")
        result_snippet = result[:300] + "..." if len(result) > 300 else result
        lines.append(
            f"[Step {n}] Thought: {thought}\n"
            f"  → Action: {action}({action_input[:120]})\n"
            f"  → Result: {result_snippet}"
        )
    return "\n\n".join(lines)


def _parse_response(content: str) -> dict[str, str]:
    """Extract JSON from LLM response, stripping any markdown fences."""
    # Strip ```json ... ``` or ``` ... ```
    clean = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()

    # Try full parse
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # Try to find first {...} block
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    return {
        "thought": str(parsed.get("thought", content[:200])),
        "action": str(parsed.get("action", "finish")),
        "action_input": str(parsed.get("action_input", content)),
    }


# ── Nodes ─────────────────────────────────────────────────────────────────────

def memory_node(state: AgentState) -> dict[str, Any]:
    """Retrieve semantically relevant memories before the first reasoning step."""
    from memory.store import MemoryStore
    from config import settings

    try:
        store = MemoryStore()
        hits = store.search(state["task"], top_k=settings.memory_top_k)
        if hits:
            ctx = "\n".join(
                f"- {h['text']} (relevance: {h['score']:.2f})"
                for h in hits
            )
        else:
            ctx = "No relevant memories found."
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")
        ctx = "Memory unavailable."

    return {"memory_context": ctx}


def reason_node(state: AgentState) -> dict[str, Any]:
    """Call Gemini to reason and choose the next action."""
    from config import settings
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()
    iteration = state.get("iteration", 0)

    system_content = SYSTEM_PROMPT.format(
        memory_context=state.get("memory_context", "None."),
        iteration=iteration,
        max_iterations=settings.max_iterations,
    )
    user_content = TASK_TEMPLATE.format(
        task=state["task"],
        history=_format_history(state.get("trace", [])),
        last_observation=state.get("tool_result", "No previous observation."),
    )

    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ])
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _parse_response(response.content)
    logger.info(
        f"[Step {iteration + 1}] action={parsed['action']} "
        f"latency={latency_ms}ms"
    )

    trace_entry = {
        "iteration": iteration + 1,
        "thought": parsed["thought"],
        "action": parsed["action"],
        "action_input": parsed["action_input"],
        "result": "",
        "latency_ms": latency_ms,
    }

    return {
        "reasoning": parsed["thought"],
        "tool_name": parsed["action"],
        "tool_input": parsed["action_input"],
        "iteration": iteration + 1,
        "trace": [trace_entry],   # Annotated list — appended automatically
    }


def act_node(state: AgentState) -> dict[str, Any]:
    """Execute the chosen tool and record the observation."""
    from tools.executor import get_tool_executor

    tool_name = state.get("tool_name", "finish")
    tool_input = state.get("tool_input", "")

    executor = get_tool_executor()
    result = executor.run(tool_name, tool_input)

    done = (tool_name == "finish")
    final_answer = result if done else ""

    # Patch the last trace entry with the result
    updated_trace = []
    existing = state.get("trace", [])
    if existing:
        last = dict(existing[-1])
        last["result"] = result
        updated_trace = [last]

    return {
        "tool_result": result,
        "final_answer": final_answer,
        "done": done,
        "trace": updated_trace,
    }


def save_memory_node(state: AgentState) -> dict[str, Any]:
    """Persist the completed task + answer to FAISS memory."""
    if not (state.get("final_answer") and state.get("task")):
        return {}

    try:
        from memory.store import MemoryStore
        store = MemoryStore()
        store.add(
            text=f"Task: {state['task']}\nAnswer: {state['final_answer'][:500]}",
            metadata={
                "task": state["task"],
                "iterations": state.get("iteration", 0),
            },
        )
        logger.info("Memory saved.")
    except Exception as e:
        logger.warning(f"Memory save failed: {e}")

    return {}


# ── Conditional edge ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route: keep looping (reason) or finish (save_memory)."""
    from config import settings
    if state.get("done"):
        return "save_memory"
    if state.get("iteration", 0) >= settings.max_iterations:
        logger.warning("Max iterations reached — forcing finish.")
        return "save_memory"
    return "reason"
