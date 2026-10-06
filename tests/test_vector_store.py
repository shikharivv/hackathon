"""Tests for src.rag.vector_store — ChromaDB vector store management."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from langchain_core.documents import Document

    from src.rag.vector_store import VectorStoreManager

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="langchain/torch not available")


@pytest.fixture()
def _sample_chunks():
    """Return a small set of synthetic documents for testing."""
    return [
        Document(
            page_content="Employees receive 15 vacation days in their first two years.",
            metadata={"source_file": "vacation.md", "policy_number": "POL-001"},
        ),
        Document(
            page_content="Primary caregivers are entitled to 16 weeks of paid parental leave.",
            metadata={"source_file": "vacation.md", "policy_number": "POL-001"},
        ),
        Document(
            page_content="The company observes 10 paid holidays per calendar year.",
            metadata={"source_file": "vacation.md", "policy_number": "POL-001"},
        ),
        Document(
            page_content="Bereavement leave provides 5 paid business days for immediate family.",
            metadata={"source_file": "vacation.md", "policy_number": "POL-001"},
        ),
        Document(
            page_content="Employees may carry over a maximum of 5 unused leave days.",
            metadata={"source_file": "vacation.md", "policy_number": "POL-001"},
        ),
    ]


@pytest.fixture()
def vector_store(tmp_path: Path, _sample_chunks, embedding_manager):
    """Create a temporary vector store populated with sample chunks."""
    emb_fn = embedding_manager.get_embedding_function()
    mgr = VectorStoreManager(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_function=emb_fn,
    )
    mgr.create_store(_sample_chunks)
    return mgr


def test_create_store(vector_store):
    """create_store should initialise the internal Chroma instance."""
    assert vector_store._store is not None  # noqa: SLF001


def test_create_store_raises_on_empty(tmp_path: Path, embedding_manager):
    """create_store should reject an empty document list."""
    emb_fn = embedding_manager.get_embedding_function()
    mgr = VectorStoreManager(
        persist_directory=tmp_path / "empty_chroma",
        collection_name="empty",
        embedding_function=emb_fn,
    )
    with pytest.raises(ValueError, match="zero documents"):
        mgr.create_store([])


def test_similarity_search_returns_results(vector_store):
    """similarity_search should return Document objects."""
    results = vector_store.similarity_search("vacation days", k=3)
    assert len(results) > 0
    assert all(isinstance(r, Document) for r in results)


def test_similarity_search_with_scores(vector_store):
    """similarity_search_with_scores should return (Document, float) tuples."""
    results = vector_store.similarity_search_with_scores("parental leave", k=2)
    assert len(results) > 0
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


def test_top_k_respected(vector_store):
    """The number of results should not exceed the requested k."""
    results = vector_store.similarity_search("leave policy", k=2)
    assert len(results) <= 2


def test_get_collection_stats(vector_store):
    """get_collection_stats should report the document count and collection name."""
    stats = vector_store.get_collection_stats()
    assert stats["collection_name"] == "test_collection"
    assert stats["document_count"] == 5
    assert "persist_directory" in stats
