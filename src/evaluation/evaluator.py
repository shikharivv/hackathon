from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

# ------------------------------------------------------------------
# Evaluation dataset
# ------------------------------------------------------------------

EVAL_QUESTIONS: list[dict[str, Any]] = [
    # --- POL-001  Vacation & Leave ---
    {
        "question": "How many vacation days do new employees get?",
        "expected_policy": "POL-001",
        "expected_answer_contains": "15 days",
    },
    {
        "question": "What is the maximum vacation accrual cap?",
        "expected_policy": "POL-001",
        "expected_answer_contains": "accrual",
    },
    {
        "question": "How far in advance must I request vacation?",
        "expected_policy": "POL-001",
        "expected_answer_contains": "advance",
    },
    {
        "question": "Is there a bereavement leave policy?",
        "expected_policy": "POL-001",
        "expected_answer_contains": "bereavement",
    },
    {
        "question": "Can I carry over unused vacation days?",
        "expected_policy": "POL-001",
        "expected_answer_contains": "carry",
    },
    # --- POL-002  Benefits ---
    {
        "question": "What is the 401(k) employer match?",
        "expected_policy": "POL-002",
        "expected_answer_contains": "6%",
    },
    {
        "question": "When does health insurance coverage start?",
        "expected_policy": "POL-002",
        "expected_answer_contains": "insurance",
    },
    {
        "question": "Are dental and vision plans included?",
        "expected_policy": "POL-002",
        "expected_answer_contains": "dental",
    },
    {
        "question": "Is there an employee tuition reimbursement programme?",
        "expected_policy": "POL-002",
        "expected_answer_contains": "tuition",
    },
    # --- POL-003  Remote Work ---
    {
        "question": "How many days per week must I work from the office?",
        "expected_policy": "POL-003",
        "expected_answer_contains": "3",
    },
    {
        "question": "Can I work remotely from another country?",
        "expected_policy": "POL-003",
        "expected_answer_contains": "remote",
    },
    {
        "question": "Do I need manager approval for remote work?",
        "expected_policy": "POL-003",
        "expected_answer_contains": "approval",
    },
    {
        "question": "What equipment is provided for remote workers?",
        "expected_policy": "POL-003",
        "expected_answer_contains": "equipment",
    },
    # --- POL-004  Onboarding / Probation ---
    {
        "question": "How long is the probation period?",
        "expected_policy": "POL-004",
        "expected_answer_contains": "90 days",
    },
    {
        "question": "What happens if I fail probation?",
        "expected_policy": "POL-004",
        "expected_answer_contains": "probation",
    },
    {
        "question": "What is included in the onboarding checklist?",
        "expected_policy": "POL-004",
        "expected_answer_contains": "onboarding",
    },
    # --- POL-005  Performance Reviews ---
    {
        "question": "When are performance reviews conducted?",
        "expected_policy": "POL-005",
        "expected_answer_contains": "June and December",
    },
    {
        "question": "Who participates in a 360-degree review?",
        "expected_policy": "POL-005",
        "expected_answer_contains": "review",
    },
    {
        "question": "What rating scale is used for performance?",
        "expected_policy": "POL-005",
        "expected_answer_contains": "rating",
    },
    {
        "question": "What happens during a Performance Improvement Plan?",
        "expected_policy": "POL-005",
        "expected_answer_contains": "improvement",
    },
    # --- POL-006  Ethics / Code of Conduct ---
    {
        "question": "What is the gift acceptance limit?",
        "expected_policy": "POL-006",
        "expected_answer_contains": "$100",
    },
    {
        "question": "How do I report an ethics violation?",
        "expected_policy": "POL-006",
        "expected_answer_contains": "report",
    },
    {
        "question": "Is there a conflict of interest disclosure requirement?",
        "expected_policy": "POL-006",
        "expected_answer_contains": "conflict",
    },
    # --- Out-of-scope (should trigger fallback) ---
    {
        "question": "What is the company stock price?",
        "expected_policy": None,
        "should_fallback": True,
    },
    {
        "question": "Can you write my resignation letter?",
        "expected_policy": None,
        "should_fallback": True,
    },
    {
        "question": "What is the weather forecast for tomorrow?",
        "expected_policy": None,
        "should_fallback": True,
    },
    {
        "question": "Help me prepare for a job interview at another company.",
        "expected_policy": None,
        "should_fallback": True,
    },
    {
        "question": "Who is the CEO of Google?",
        "expected_policy": None,
        "should_fallback": True,
    },
    # --- Ambiguous / edge-case questions ---
    {
        "question": "What are my rights?",
        "expected_policy": None,
        "expected_answer_contains": None,
    },
    {
        "question": "Tell me everything about policies.",
        "expected_policy": None,
        "expected_answer_contains": None,
    },
]


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class RAGEvaluator:
    """End-to-end evaluation harness for the HR Policy RAG pipeline."""

    def __init__(self, rag_chain: Any) -> None:
        self.rag_chain = rag_chain

    # ---------------------------------------------------------------
    # Single evaluation
    # ---------------------------------------------------------------
    def evaluate_single(
        self,
        question: str,
        expected_answer: str | None = None,
        expected_policy: str | None = None,
    ) -> dict[str, Any]:
        """Run a single query and return structured evaluation metrics."""

        start = time.perf_counter()
        try:
            result: dict[str, Any] = self.rag_chain.query(question=question)
        except Exception:
            logger.exception("Evaluation query failed: {}", question)
            return {
                "question": question,
                "answer": "",
                "latency_ms": _elapsed_ms(start),
                "sources_found": 0,
                "correct_policy_cited": False,
                "confidence_score": 0.0,
                "fallback_triggered": True,
                "error": True,
            }

        latency_ms = _elapsed_ms(start)
        sources = result.get("sources", [])
        cited_policies = [s.get("policy_number") for s in sources]

        correct_policy_cited = (
            expected_policy in cited_policies if expected_policy else False
        )

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "latency_ms": latency_ms,
            "sources_found": len(sources),
            "correct_policy_cited": correct_policy_cited,
            "confidence_score": float(result.get("confidence", 0.0)),
            "fallback_triggered": result.get("fallback_triggered", False),
            "error": False,
        }

    # ---------------------------------------------------------------
    # Batch evaluation
    # ---------------------------------------------------------------
    def evaluate_batch(
        self, questions: list[dict[str, Any]] | None = None
    ) -> pd.DataFrame:
        """Evaluate a list of questions and return a DataFrame of results.

        If *questions* is ``None`` the built-in :data:`EVAL_QUESTIONS` are used.
        """

        dataset = questions or EVAL_QUESTIONS
        rows: list[dict[str, Any]] = []

        for idx, item in enumerate(dataset, start=1):
            q = item["question"]
            logger.info("[{}/{}] Evaluating: {}", idx, len(dataset), q)
            row = self.evaluate_single(
                question=q,
                expected_answer=item.get("expected_answer_contains"),
                expected_policy=item.get("expected_policy"),
            )
            row["expected_policy"] = item.get("expected_policy")
            row["should_fallback"] = item.get("should_fallback", False)
            rows.append(row)

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------
    @staticmethod
    def compute_metrics(results: pd.DataFrame) -> dict[str, float]:
        """Derive aggregate quality metrics from evaluation *results*."""

        in_scope = results[results["expected_policy"].notna()]
        out_of_scope = results[results["should_fallback"] == True]  # noqa: E712

        total = len(results)
        accuracy = (
            in_scope["correct_policy_cited"].mean() if len(in_scope) else 0.0
        )
        fallback_rate = (
            results["fallback_triggered"].mean() if total else 0.0
        )
        correct_fallback_rate = (
            out_of_scope["fallback_triggered"].mean()
            if len(out_of_scope)
            else 0.0
        )

        return {
            "total_questions": total,
            "accuracy": round(float(accuracy), 4),
            "fallback_rate": round(float(fallback_rate), 4),
            "correct_fallback_rate": round(float(correct_fallback_rate), 4),
            "avg_latency_ms": round(float(results["latency_ms"].mean()), 2),
            "avg_confidence": round(
                float(results["confidence_score"].mean()), 4
            ),
            "avg_sources_per_query": round(
                float(results["sources_found"].mean()), 2
            ),
        }

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    def generate_report(self, results: pd.DataFrame) -> str:
        """Produce a human-readable evaluation report."""

        metrics = self.compute_metrics(results)

        lines: list[str] = [
            "=" * 60,
            "  HR Policy RAG — Evaluation Report",
            "=" * 60,
            "",
            f"  Total questions evaluated : {metrics['total_questions']}",
            f"  Policy-citation accuracy  : {metrics['accuracy']:.2%}",
            f"  Fallback rate (overall)   : {metrics['fallback_rate']:.2%}",
            f"  Correct fallback rate     : {metrics['correct_fallback_rate']:.2%}",
            f"  Average latency           : {metrics['avg_latency_ms']:.0f} ms",
            f"  Average confidence        : {metrics['avg_confidence']:.2%}",
            f"  Avg sources per query     : {metrics['avg_sources_per_query']:.1f}",
            "",
            "-" * 60,
            "  Per-question breakdown",
            "-" * 60,
        ]

        for _, row in results.iterrows():
            status = "PASS" if row.get("correct_policy_cited") else "FAIL"
            if row.get("should_fallback"):
                status = (
                    "PASS (fallback)"
                    if row.get("fallback_triggered")
                    else "FAIL (no fallback)"
                )
            lines.append(
                f"  [{status:>16}] {row['question'][:60]}"
            )

        lines.append("")
        lines.append("=" * 60)
        report = "\n".join(lines)
        logger.info("\n{}", report)
        return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
