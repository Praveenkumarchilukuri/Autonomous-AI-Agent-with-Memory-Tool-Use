"""
Embedder — Gemini text-embedding-004 with a deterministic fallback.

When GEMINI_API_KEY is absent (CI, tests, offline dev) we produce a
deterministic 768-dim vector from MD5 + random projection so that all
other code (FAISS, cosine search, etc.) works without any API key.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class GeminiEmbedder:
    """Wraps the Gemini embedding model with a numpy interface."""

    DIMENSION = 768  # text-embedding-004 output dimension

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from config import settings
        api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_embedding_model

        self._use_api = False
        if _GENAI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._use_api = True
            except Exception as e:
                logger.warning(f"Gemini configure failed ({e}); using fallback embedder.")

    def embed(self, text: str) -> np.ndarray:
        """Return a (768,) float32 L2-normalised embedding."""
        if self._use_api:
            try:
                return self._embed_api(text)
            except Exception as e:
                logger.warning(f"Gemini embed failed ({e}); falling back.")
        return self._embed_fallback(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return (N, 768) float32 matrix."""
        return np.vstack([self.embed(t) for t in texts])

    # ── Private ──────────────────────────────────────────────────────

    def _embed_api(self, text: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document",
        )
        vec = np.array(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def _embed_fallback(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding — consistent across runs."""
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)
