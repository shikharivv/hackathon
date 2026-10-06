"""Tests for the RAG chain orchestration."""

from __future__ import annotations

import pytest

try:
    from langchain_core.documents import Document  # noqa: F401

    from src.rag.chain import HRPolicyRAGChain  # noqa: F401

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="langchain/torch not available")


def test_citation_dataclass():
    """Citation dataclass should hold expected fields."""
    from src.rag.chain import Citation

    c = Citation(
        policy_number="POL-001",
        policy_title="Vacation",
        section="Section 3",
        chunk_text="15 days of leave",
        relevance_score=0.85,
    )
    assert c.policy_number == "POL-001"
    assert c.relevance_score == 0.85


def test_rag_response_dataclass():
    """RAGResponse dataclass should hold expected fields."""
    from src.rag.chain import RAGResponse

    r = RAGResponse(
        answer="15 days",
        citations=[],
        confidence=0.9,
        sources_used=1,
        fallback_triggered=False,
        guardrail_flags=[],
        latency_ms=120.5,
    )
    assert r.answer == "15 days"
    assert r.fallback_triggered is False


def test_guardrail_result_dataclass():
    """GuardrailResult should be importable from guardrails module."""
    from src.rag.guardrails import GuardrailResult

    gr = GuardrailResult(passed=True, flags=[])
    assert gr.passed is True
    assert gr.flags == []
