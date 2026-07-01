"""Unit tests for FAISS memory store — no API key required (uses fallback embedder)."""

import tempfile
import numpy as np
import pytest

from memory.embedder import GeminiEmbedder
from memory.store import MemoryStore


class TestGeminiEmbedder:
    def test_shape(self):
        e = GeminiEmbedder()
        v = e.embed("hello world")
        assert v.shape == (768,)
        assert v.dtype == np.float32

    def test_normalised(self):
        e = GeminiEmbedder()
        v = e.embed("normalisation test")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_deterministic(self):
        e = GeminiEmbedder()
        assert np.allclose(e.embed("foo"), e.embed("foo"))

    def test_different_texts_differ(self):
        e = GeminiEmbedder()
        assert not np.allclose(e.embed("cats"), e.embed("quantum physics"))

    def test_batch(self):
        e = GeminiEmbedder()
        mat = e.embed_batch(["a", "b", "c"])
        assert mat.shape == (3, 768)


class TestMemoryStore:
    def test_add_and_len(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            assert len(s) == 0
            s.add("hello memory")
            assert len(s) == 1

    def test_search_returns_results(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            s.add("The capital of France is Paris", metadata={"answer": "Paris"})
            s.add("Python is a programming language")
            results = s.search("France capital", top_k=2)
            assert len(results) >= 1
            assert "text" in results[0]
            assert "score" in results[0]

    def test_empty_search(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            assert s.search("anything") == []

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            s.add("persist me", metadata={"tag": "test"})
            # Force save
            s._save()

            s2 = MemoryStore(d)
            assert len(s2) == 1
            assert s2._metadata[0]["text"] == "persist me"

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            s.add("will be cleared")
            s.clear()
            assert len(s) == 0
            assert s.search("cleared") == []

    def test_top_k_limit(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            for i in range(10):
                s.add(f"Entry {i}")
            results = s.search("entry", top_k=3)
            assert len(results) <= 3

    def test_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            s = MemoryStore(d)
            s.add("meta test", metadata={"key": "value"})
            results = s.search("meta test", top_k=1)
            assert results[0]["metadata"]["key"] == "value"
