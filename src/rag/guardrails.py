from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class GuardrailResult:
    """Aggregated outcome of all guardrail checks."""

    passed: bool
    flags: list[str] = field(default_factory=list)


class ResponseGuardrails:
    """Anti-hallucination guardrails for RAG responses.

    Every public method is a self-contained check that can be composed via
    :meth:`apply_all`.
    """

    BLOCKED_PATTERNS: list[re.Pattern[str]] = [
        # Salary / compensation amounts without explicit source reference
        re.compile(
            r"\$\s*[\d,]+(?:\.\d{2})?\s*(?:per|/)\s*(?:year|month|hour|annum)",
            re.IGNORECASE,
        ),
        # Legal advice indicators
        re.compile(
            r"\b(?:you\s+should\s+(?:sue|file\s+a\s+lawsuit|consult\s+(?:a|your)\s+lawyer))\b",
            re.IGNORECASE,
        ),
        # Medical advice indicators
        re.compile(
            r"\b(?:you\s+should\s+(?:take|stop\s+taking)\s+(?:medication|medicine))\b",
            re.IGNORECASE,
        ),
        # Definitive legal conclusions
        re.compile(
            r"\b(?:this\s+(?:is|constitutes)\s+(?:illegal|a\s+violation\s+of\s+law))\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, confidence_threshold: float = 0.65) -> None:
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_source_grounding(
        self, response: str, source_texts: list[str]
    ) -> dict[str, object]:
        """Verify that key factual claims in the *response* appear in *source_texts*.

        Uses a simple sentence-overlap heuristic: each response sentence is
        considered *grounded* if at least one substantial n-gram (4+ words)
        appears verbatim in any source chunk.
        """

        sentences = [
            s.strip()
            for s in re.split(r"[.!?]\s+", response)
            if len(s.strip()) > 20
        ]

        if not sentences:
            return {"grounded": True, "ungrounded_claims": []}

        combined_sources = " ".join(source_texts).lower()
        ungrounded: list[str] = []

        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) < 4:
                continue

            grounded = False
            # Sliding window of 4-grams
            for i in range(len(words) - 3):
                ngram = " ".join(words[i : i + 4])
                if ngram in combined_sources:
                    grounded = True
                    break

            if not grounded:
                ungrounded.append(sentence)

        is_grounded = len(ungrounded) <= len(sentences) * 0.5

        if not is_grounded:
            logger.warning(
                "Grounding check failed: {}/{} sentences ungrounded",
                len(ungrounded),
                len(sentences),
            )

        return {"grounded": is_grounded, "ungrounded_claims": ungrounded}

    def check_confidence(
        self, scores: list[float]
    ) -> dict[str, object]:
        """Check whether retrieval similarity scores meet the threshold."""

        if not scores:
            return {
                "sufficient": False,
                "avg_score": 0.0,
                "min_score": 0.0,
            }

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)

        sufficient = avg_score >= self.confidence_threshold

        if not sufficient:
            logger.debug(
                "Confidence check: avg={:.3f} < threshold={:.3f}",
                avg_score,
                self.confidence_threshold,
            )

        return {
            "sufficient": sufficient,
            "avg_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
        }

    def check_refusal_needed(
        self, scores: list[float], response: str
    ) -> bool:
        """Return ``True`` when the system should refuse to answer.

        A refusal is triggered when:
        * No retrieval scores are available, **or**
        * Average similarity is below the confidence threshold, **or**
        * The response matches a blocked pattern.
        """

        if not scores:
            return True

        avg_score = sum(scores) / len(scores)
        if avg_score < self.confidence_threshold:
            return True

        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(response):
                logger.warning(
                    "Blocked pattern detected in response: {}", pattern.pattern
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def apply_all(
        self,
        response: str,
        source_texts: list[str],
        scores: list[float],
    ) -> GuardrailResult:
        """Run **all** guardrail checks and return an aggregated result."""

        flags: list[str] = []

        # 1. Confidence
        confidence = self.check_confidence(scores)
        if not confidence["sufficient"]:
            flags.append("low_confidence")

        # 2. Source grounding
        grounding = self.check_source_grounding(response, source_texts)
        if not grounding["grounded"]:
            flags.append("potential_hallucination")

        # 3. Refusal / blocked patterns
        if not scores:
            flags.append("no_sources_found")
        else:
            for pattern in self.BLOCKED_PATTERNS:
                if pattern.search(response):
                    flags.append("blocked_pattern_detected")
                    break

        passed = len(flags) == 0

        logger.debug(
            "Guardrail result: passed={}, flags={}", passed, flags
        )
        return GuardrailResult(passed=passed, flags=flags)
