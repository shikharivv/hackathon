"""Tests for src.rag.document_loader — policy loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from langchain_core.documents import Document

    from src.rag.document_loader import PolicyDocumentLoader

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="langchain/torch not available")

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_documents_count(policies_dir: Path):
    """The loader should find all .md policy files under the policies dir."""
    loader = PolicyDocumentLoader(policies_dir)
    docs = loader.load_documents()
    assert len(docs) >= 1


def test_document_has_metadata(raw_documents):
    """Every loaded document should carry source_file and title metadata."""
    for doc in raw_documents:
        assert "source_file" in doc.metadata
        assert "title" in doc.metadata


def test_document_has_policy_number(raw_documents):
    """Documents with an explicit Policy Number header should parse it."""
    has_policy_number = any("policy_number" in doc.metadata for doc in raw_documents)
    assert has_policy_number, "At least one document should have a policy_number"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_documents_smaller_than_original(raw_documents):
    """Chunked output should contain more items than the original documents."""
    loader = PolicyDocumentLoader(Path("."), chunk_size=512, chunk_overlap=64)
    chunks = loader.chunk_documents(raw_documents)
    assert len(chunks) > len(raw_documents)


def test_chunks_preserve_metadata(raw_documents):
    """Every chunk should retain the source document metadata."""
    loader = PolicyDocumentLoader(Path("."), chunk_size=512, chunk_overlap=64)
    chunks = loader.chunk_documents(raw_documents)
    for chunk in chunks:
        assert "source_file" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_load_and_chunk_returns_documents(policies_dir: Path):
    """load_and_chunk should return a non-empty list of Document objects."""
    loader = PolicyDocumentLoader(policies_dir)
    chunks = loader.load_and_chunk()
    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_directory_returns_empty(tmp_path: Path):
    """An empty directory should yield an empty list (no crash)."""
    loader = PolicyDocumentLoader(tmp_path)
    docs = loader.load_documents()
    assert docs == []


def test_nonexistent_directory_returns_empty():
    """A non-existent path should yield an empty list (no crash)."""
    loader = PolicyDocumentLoader(Path("/does/not/exist"))
    docs = loader.load_documents()
    assert docs == []
