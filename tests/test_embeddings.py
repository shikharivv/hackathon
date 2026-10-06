"""Tests for src.rag.embeddings — sentence-transformer embedding manager."""

from __future__ import annotations

import math

import pytest

try:
    from src.rag.embeddings import EmbeddingManager  # noqa: F401

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="langchain/torch not available")

# ---------------------------------------------------------------------------
# Single text embedding
# ---------------------------------------------------------------------------


def test_embed_text_returns_vector(embedding_manager):
    """embed_text should return a list of floats."""
    vector = embedding_manager.embed_text("What is the vacation policy?")
    assert isinstance(vector, list)
    assert all(isinstance(v, float) for v in vector)


def test_embed_text_dimension(embedding_manager):
    """MiniLM-L6-v2 produces 384-dimensional vectors."""
    vector = embedding_manager.embed_text("Employee leave request")
    assert len(vector) == 384


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


def test_embed_texts_batch(embedding_manager):
    """embed_texts should return one vector per input text."""
    texts = ["Annual leave entitlement", "Parental leave duration", "Holiday pay rate"]
    vectors = embedding_manager.embed_texts(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == 384 for v in vectors)


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_similar_texts_close_vectors(embedding_manager):
    """Semantically similar texts should have higher cosine similarity."""
    v_leave = embedding_manager.embed_text("How many vacation days do I get?")
    v_similar = embedding_manager.embed_text("What is my annual leave entitlement?")
    v_unrelated = embedding_manager.embed_text("What is the chemical formula for water?")

    sim_close = _cosine_similarity(v_leave, v_similar)
    sim_far = _cosine_similarity(v_leave, v_unrelated)

    assert sim_close > sim_far
