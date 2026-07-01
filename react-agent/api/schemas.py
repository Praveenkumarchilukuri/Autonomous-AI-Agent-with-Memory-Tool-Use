"""Pydantic request / response models for the FastAPI layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000,
                      description="The task for the agent to solve.")
    max_steps: int = Field(default=15, ge=1, le=50)


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


# ── Responses ─────────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    iteration: int
    thought: str
    action: str
    action_input: str
    result: str
    latency_ms: int


class AgentResponse(BaseModel):
    task: str
    final_answer: str | None
    trace: list[TraceStep]
    iterations: int
    success: bool


class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    memory_size: int
    model: str
    tools: list[str]
