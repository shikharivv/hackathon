"""Shared fixtures for the HR Policy RAG Assistant test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

try:

    _LANGCHAIN_AVAILABLE = True
except Exception:
    _LANGCHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def policies_dir(project_root: Path) -> Path:
    """Return the path to the HR policy documents directory."""
    return project_root / "docs" / "policies"


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_documents(policies_dir: Path) -> list[Any]:
    """Load and chunk all policy documents from disk."""
    if not _LANGCHAIN_AVAILABLE:
        pytest.skip("langchain not available")
    from src.rag.document_loader import PolicyDocumentLoader

    loader = PolicyDocumentLoader(policies_dir)
    return loader.load_and_chunk()


@pytest.fixture()
def raw_documents(policies_dir: Path) -> list[Any]:
    """Load policy documents without chunking."""
    if not _LANGCHAIN_AVAILABLE:
        pytest.skip("langchain not available")
    from src.rag.document_loader import PolicyDocumentLoader

    loader = PolicyDocumentLoader(policies_dir)
    return loader.load_documents()


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedding_manager() -> Any:
    """Return a shared EmbeddingManager instance."""
    if not _LANGCHAIN_AVAILABLE:
        pytest.skip("langchain not available")
    try:
        from src.rag.embeddings import EmbeddingManager

        mgr = EmbeddingManager()
        # Force load to detect torch issues early
        mgr.get_embedding_function()
        return mgr
    except Exception as exc:
        pytest.skip(f"Embedding model not available: {exc}")


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_response() -> MagicMock:
    """Return a mock LLM that produces a deterministic answer."""
    mock = MagicMock()
    mock.invoke.return_value = (
        "According to company policy, employees receive 15 days "
        "of annual leave during their first two years."
    )
    return mock


@pytest.fixture()
def sample_query() -> str:
    """Return a representative HR policy question."""
    return "How many vacation days does a new employee get?"
