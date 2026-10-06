"""Tests for the RAG evaluation pipeline (src.evaluation.evaluator)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.evaluation.evaluator import EVAL_QUESTIONS, RAGEvaluator

# ---------------------------------------------------------------------------
# Evaluation question set
# ---------------------------------------------------------------------------


def test_eval_questions_non_empty():
    """The built-in question set should contain entries."""
    assert isinstance(EVAL_QUESTIONS, list)
    assert len(EVAL_QUESTIONS) > 0


def test_eval_questions_have_required_keys():
    """Each question dict should have at minimum a 'question' key."""
    for item in EVAL_QUESTIONS:
        assert "question" in item


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------


def test_evaluate_single_returns_dict():
    """evaluate_single should return a dict with expected keys."""
    mock_chain = MagicMock()
    mock_chain.query.return_value = {
        "answer": "15 days",
        "sources": [{"policy_number": "POL-001"}],
        "confidence": 0.9,
        "fallback_triggered": False,
    }
    evaluator = RAGEvaluator(rag_chain=mock_chain)
    result = evaluator.evaluate_single(
        question="How many vacation days?",
        expected_answer="15 days",
        expected_policy="POL-001",
    )
    assert isinstance(result, dict)
    assert "question" in result
    assert "latency_ms" in result
    assert "correct_policy_cited" in result
    assert result["correct_policy_cited"] is True


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_keys():
    """compute_metrics should return standard evaluation metric keys."""
    df = pd.DataFrame(
        [
            {
                "question": "Q1",
                "correct_policy_cited": True,
                "fallback_triggered": False,
                "confidence_score": 0.9,
                "latency_ms": 100.0,
                "sources_found": 3,
                "expected_policy": "POL-001",
                "should_fallback": False,
            },
            {
                "question": "Q2",
                "correct_policy_cited": False,
                "fallback_triggered": True,
                "confidence_score": 0.3,
                "latency_ms": 50.0,
                "sources_found": 0,
                "expected_policy": None,
                "should_fallback": True,
            },
        ]
    )
    metrics = RAGEvaluator.compute_metrics(df)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "avg_latency_ms" in metrics
