"""Central settings — reads from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    # Gemini
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-pro"))
    gemini_temperature: float = field(default_factory=lambda: float(_env("AGENT_TEMPERATURE", "0.1")))
    gemini_max_tokens: int = field(default_factory=lambda: int(_env("GEMINI_MAX_TOKENS", "2048")))
    gemini_embedding_model: str = field(default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))

    # Tools
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))

    # Memory
    faiss_index_path: str = field(default_factory=lambda: _env("FAISS_PERSIST_PATH", "./faiss_store"))
    memory_top_k: int = field(default_factory=lambda: int(_env("MEMORY_TOP_K", "5")))
    memory_score_threshold: float = field(default_factory=lambda: float(_env("MEMORY_SCORE_THRESHOLD", "0.70")))
    embedding_dim: int = 768  # Gemini text-embedding-004 output dimension

    # Agent
    max_iterations: int = field(default_factory=lambda: int(_env("AGENT_MAX_STEPS", "15")))

    # API
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_env("API_PORT", "8000")))
    api_log_level: str = field(default_factory=lambda: _env("API_LOG_LEVEL", "info"))

    # Gradio
    gradio_host: str = field(default_factory=lambda: _env("GRADIO_HOST", "0.0.0.0"))
    gradio_port: int = field(default_factory=lambda: int(_env("GRADIO_PORT", "7860")))
    gradio_share: bool = field(default_factory=lambda: _env("GRADIO_SHARE", "false").lower() == "true")


# Load .env if present (optional dependency)
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
settings = Settings()
