"""Memory package — FAISS vector store + Gemini embedder."""

from memory.embedder import GeminiEmbedder
from memory.store import MemoryStore

__all__ = ["GeminiEmbedder", "MemoryStore"]
