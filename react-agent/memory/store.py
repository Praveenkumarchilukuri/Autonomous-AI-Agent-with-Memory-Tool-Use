"""
Persistent FAISS vector memory store.

  store = MemoryStore()
  store.add("Task: GDP France 2024", metadata={"answer": "~$3.1T"})

  # Later session
  results = store.search("economic data France", top_k=3)
  # [{"text": "Task: GDP France 2024", "metadata": {...}, "score": 0.91}]
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False

from memory.embedder import GeminiEmbedder

logger = logging.getLogger(__name__)


class MemoryStore:
    """FAISS-backed persistent memory with automatic fallback to pure numpy."""

    _INDEX_FILE = "index.faiss"
    _META_FILE = "metadata.json"
    _VEC_FILE = "vectors.npy"          # numpy fallback

    def __init__(self, index_path: str | None = None) -> None:
        from config import settings
        self.path = Path(index_path or settings.faiss_index_path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.embedder = GeminiEmbedder()
        self.dim = GeminiEmbedder.DIMENSION

        self._metadata: list[dict[str, Any]] = []
        self._index = None          # faiss index  or None
        self._vecs: list[np.ndarray] = []   # numpy fallback

        self._load()

    # ── Public API ────────────────────────────────────────────────────

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Embed text and store. Returns entry index."""
        vec = self.embedder.embed(text).reshape(1, -1)
        self._metadata.append({"text": text, "metadata": metadata or {}})

        if _FAISS and self._index is not None:
            self._index.add(vec)
        else:
            self._vecs.append(vec)

        self._save()
        idx = len(self._metadata) - 1
        logger.debug(f"Memory +{idx}: {text[:80]}")
        return idx

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {text, metadata, score} dicts."""
        if not self._metadata:
            return []

        q_vec = self.embedder.embed(query).reshape(1, -1)
        k = min(top_k, len(self._metadata))

        if _FAISS and self._index is not None and self._index.ntotal > 0:
            scores, indices = self._index.search(q_vec, k)
            return [
                {
                    "text": self._metadata[i]["text"],
                    "metadata": self._metadata[i]["metadata"],
                    "score": float(s),
                }
                for s, i in zip(scores[0], indices[0])
                if i >= 0
            ]

        # Numpy fallback
        if not self._vecs:
            return []
        mat = np.vstack(self._vecs)           # (N, D)
        sims = (mat @ q_vec.T).squeeze()      # (N,)
        if sims.ndim == 0:
            sims = sims.reshape(1)
        top_idx = np.argsort(-sims)[:k]
        return [
            {
                "text": self._metadata[i]["text"],
                "metadata": self._metadata[i]["metadata"],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]

    def clear(self) -> None:
        """Wipe all memories from disk and RAM."""
        self._metadata = []
        self._vecs = []
        self._init_index()
        self._save()
        logger.info("Memory store cleared.")

    @property
    def size(self) -> int:
        return len(self._metadata)

    def __len__(self) -> int:
        return self.size

    # ── Persistence ───────────────────────────────────────────────────

    def _init_index(self) -> None:
        if _FAISS:
            self._index = faiss.IndexFlatIP(self.dim)

    def _load(self) -> None:
        meta_file = self.path / self._META_FILE

        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    self._metadata = json.load(f)

                if _FAISS:
                    faiss_file = self.path / self._INDEX_FILE
                    if faiss_file.exists():
                        self._index = faiss.read_index(str(faiss_file))
                        logger.info(f"Loaded {self._index.ntotal} vectors from FAISS")
                        return
                    # Rebuild FAISS from numpy if available
                    vec_file = self.path / self._VEC_FILE
                    if vec_file.exists():
                        vecs = np.load(str(vec_file))
                        self._init_index()
                        self._index.add(vecs)
                        logger.info(f"Rebuilt FAISS from {len(vecs)} stored vectors")
                        return

                # Numpy fallback load
                vec_file = self.path / self._VEC_FILE
                if vec_file.exists():
                    mat = np.load(str(vec_file))
                    self._vecs = [mat[i:i+1] for i in range(len(mat))]
                    logger.info(f"Loaded {len(self._vecs)} vectors (numpy mode)")
                    return
            except Exception as e:
                logger.warning(f"Memory load failed ({e}). Starting fresh.")

        self._metadata = []
        self._init_index()

    def _save(self) -> None:
        try:
            with open(self.path / self._META_FILE, "w") as f:
                json.dump(self._metadata, f, indent=2)

            if _FAISS and self._index is not None:
                faiss.write_index(self._index, str(self.path / self._INDEX_FILE))
            elif self._vecs:
                np.save(str(self.path / self._VEC_FILE), np.vstack(self._vecs))
        except Exception as e:
            logger.error(f"Memory save failed: {e}")
