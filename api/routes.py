"""FastAPI route handlers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    TraceStep,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_to_response(state: dict, task: str) -> AgentResponse:
    steps = [TraceStep(**s) for s in state.get("trace", [])]
    return AgentResponse(
        task=task,
        final_answer=state.get("final_answer") or None,
        trace=steps,
        iterations=state.get("iteration", 0),
        success=bool(state.get("final_answer")),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    from memory.store import MemoryStore
    from config import settings

    store = MemoryStore()
    return HealthResponse(
        status="ok",
        memory_size=store.size,
        model=settings.gemini_model,
        tools=["web_search", "python_repl", "wikipedia", "finish"],
    )


# ── Agent — synchronous run ────────────────────────────────────────────────────

@router.post("/agent/run", response_model=AgentResponse)
async def agent_run(req: AgentRequest):
    """Run the agent synchronously and return the complete result."""
    from agent.graph import run_agent

    loop = asyncio.get_event_loop()
    try:
        t0 = time.perf_counter()
        state = await loop.run_in_executor(None, lambda: run_agent(req.task))
        elapsed = time.perf_counter() - t0
        logger.info(f"Agent finished '{req.task[:50]}' in {elapsed:.2f}s")
        return _state_to_response(state, req.task)
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent — SSE streaming ──────────────────────────────────────────────────────

@router.post("/agent/stream")
async def agent_stream(req: AgentRequest):
    """
    Stream agent reasoning steps via Server-Sent Events.

    Each event is JSON with fields: node, data.
    Final event has node='done' with the full answer.
    """
    from agent.graph import stream_agent

    async def generator() -> AsyncGenerator[dict, None]:
        try:
            loop = asyncio.get_event_loop()
            # stream_agent is a sync generator — run it in executor
            queue: asyncio.Queue = asyncio.Queue()
            done_sentinel = object()

            def _produce():
                try:
                    for chunk in stream_agent(req.task):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, done_sentinel)

            loop.run_in_executor(None, _produce)

            final_state = {}
            while True:
                chunk = await queue.get()
                if chunk is done_sentinel:
                    break
                for node_name, node_state in chunk.items():
                    final_state.update(node_state)
                    yield {
                        "event": "step",
                        "data": json.dumps({
                            "node": node_name,
                            "iteration": node_state.get("iteration", 0),
                            "action": node_state.get("tool_name", ""),
                            "thought": node_state.get("reasoning", ""),
                            "result": node_state.get("tool_result", ""),
                        }),
                    }

            yield {
                "event": "done",
                "data": json.dumps({
                    "final_answer": final_state.get("final_answer", ""),
                    "iterations": final_state.get("iteration", 0),
                    "success": bool(final_state.get("final_answer")),
                }),
            }

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generator())


# ── Memory ────────────────────────────────────────────────────────────────────

@router.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest):
    from memory.store import MemoryStore
    store = MemoryStore()
    results = store.search(req.query, top_k=req.k)
    return MemorySearchResponse(query=req.query, results=results, count=len(results))


@router.delete("/memory/clear")
async def memory_clear():
    from memory.store import MemoryStore
    store = MemoryStore()
    store.clear()
    return {"status": "cleared", "size": 0}


@router.get("/memory/size")
async def memory_size():
    from memory.store import MemoryStore
    return {"size": MemoryStore().size}
