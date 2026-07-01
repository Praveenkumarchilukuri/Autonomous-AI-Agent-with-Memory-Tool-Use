"""
FastAPI application.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.api_log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Autonomous ReAct Agent API",
    description=(
        "LangGraph + Gemini 1.5 Pro agent with FAISS persistent memory, "
        "multi-tool execution (web search, Python REPL, Wikipedia), "
        "and live SSE streaming of reasoning chains."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def latency_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    return response


@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logging.getLogger(__name__).exception(f"Unhandled error on {request.url}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


from api.routes import router  # noqa: E402
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.api_log_level,
        reload=True,
    )
