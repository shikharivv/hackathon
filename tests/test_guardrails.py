"""Tests for guardrails logic (src.rag.guardrails)."""

from __future__ import annotations

from src.rag.guardrails import GuardrailResult, ResponseGuardrails

# ---------------------------------------------------------------------------
# Confidence checks
# ---------------------------------------------------------------------------


def test_check_confidence_sufficient():
    """A score above the threshold should pass."""
    g = ResponseGuardrails(confidence_threshold=0.65)
    result = g.check_confidence([0.85, 0.90, 0.78])
    assert result["sufficient"] is True


def test_check_confidence_insufficient():
    """A score below the threshold should fail."""
    g = ResponseGuardrails(confidence_threshold=0.65)
    result = g.check_confidence([0.30, 0.20, 0.40])
    assert result["sufficient"] is False


# ---------------------------------------------------------------------------
# Source grounding
# ---------------------------------------------------------------------------


def test_check_source_grounding_grounded():
    """An answer well-supported by source text should be flagged as grounded."""
    g = ResponseGuardrails()
    result = g.check_source_grounding(
        response="Employees receive 15 vacation days in their first two years.",
        source_texts=["Employees receive 15 vacation days in their first two years of employment."],
    )
    assert result["grounded"] is True


def test_check_source_grounding_ungrounded():
    """An answer with no source overlap should be flagged as ungrounded."""
    g = ResponseGuardrails()
    result = g.check_source_grounding(
        response="The moon is made of green cheese and employees love it.",
        source_texts=["Employees receive 15 vacation days in their first two years."],
    )
    assert result["grounded"] is False


# ---------------------------------------------------------------------------
# Refusal check
# ---------------------------------------------------------------------------


def test_check_refusal_needed_low_scores():
    """When scores are very low, refusal should be triggered."""
    g = ResponseGuardrails(confidence_threshold=0.65)
    should_refuse = g.check_refusal_needed(
        scores=[0.10, 0.15, 0.12],
        response="Some ungrounded answer.",
    )
    assert should_refuse is True


def test_check_refusal_not_needed_high_scores():
    """When scores are high, no refusal needed."""
    g = ResponseGuardrails(confidence_threshold=0.65)
    should_refuse = g.check_refusal_needed(
        scores=[0.85, 0.90, 0.78],
        response="Well-grounded answer.",
    )
    assert should_refuse is False


# ---------------------------------------------------------------------------
# Composite guard
# ---------------------------------------------------------------------------


def test_apply_all_passes():
    """apply_all should return a GuardrailResult."""
    g = ResponseGuardrails(confidence_threshold=0.65)
    result = g.apply_all(
        response="Employees receive 15 vacation days.",
        source_texts=["Employees receive 15 vacation days in their first two years."],
        scores=[0.90, 0.85],
    )
    assert isinstance(result, GuardrailResult)
    assert isinstance(result.flags, list)
