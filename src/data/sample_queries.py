from __future__ import annotations

from typing import Any

SAMPLE_QUERIES: list[dict[str, Any]] = [
    # --- Vacation & Leave (easy) ---
    {
        "question": "How many vacation days do I get after 3 years?",
        "category": "vacation",
        "difficulty": "easy",
    },
    {
        "question": "Can I carry over unused vacation days to the next year?",
        "category": "vacation",
        "difficulty": "easy",
    },
    {
        "question": "How do I request time off?",
        "category": "vacation",
        "difficulty": "easy",
    },
    {
        "question": "What holidays does the company observe?",
        "category": "vacation",
        "difficulty": "easy",
    },
    # --- Benefits (easy / medium) ---
    {
        "question": "What is the 401(k) matching policy?",
        "category": "benefits",
        "difficulty": "easy",
    },
    {
        "question": "When am I eligible for health insurance?",
        "category": "benefits",
        "difficulty": "easy",
    },
    {
        "question": "Does the company offer life insurance?",
        "category": "benefits",
        "difficulty": "medium",
    },
    {
        "question": "Is there a tuition reimbursement programme?",
        "category": "benefits",
        "difficulty": "medium",
    },
    # --- Remote Work (easy / medium) ---
    {
        "question": "Can I work remotely from another country?",
        "category": "remote_work",
        "difficulty": "medium",
    },
    {
        "question": "How many days per week must I be in the office?",
        "category": "remote_work",
        "difficulty": "easy",
    },
    {
        "question": "What equipment is provided for remote workers?",
        "category": "remote_work",
        "difficulty": "easy",
    },
    {
        "question": "Do I need approval for a hybrid schedule change?",
        "category": "remote_work",
        "difficulty": "medium",
    },
    # --- Performance Reviews (medium) ---
    {
        "question": "What happens during a Performance Improvement Plan?",
        "category": "performance",
        "difficulty": "medium",
    },
    {
        "question": "When are performance reviews conducted?",
        "category": "performance",
        "difficulty": "easy",
    },
    {
        "question": "What criteria are used in annual performance evaluations?",
        "category": "performance",
        "difficulty": "medium",
    },
    {
        "question": "How does the 360-degree feedback process work?",
        "category": "performance",
        "difficulty": "hard",
    },
    # --- Ethics / Code of Conduct (medium / hard) ---
    {
        "question": "What is the gift policy limit?",
        "category": "ethics",
        "difficulty": "easy",
    },
    {
        "question": "How do I report an ethics violation?",
        "category": "ethics",
        "difficulty": "medium",
    },
    {
        "question": "What constitutes a conflict of interest?",
        "category": "ethics",
        "difficulty": "hard",
    },
    {
        "question": "Is there a whistleblower protection policy?",
        "category": "ethics",
        "difficulty": "hard",
    },
    # --- Onboarding / Probation (easy / medium) ---
    {
        "question": "How long is the probation period for new hires?",
        "category": "onboarding",
        "difficulty": "easy",
    },
    {
        "question": "What does the onboarding process include?",
        "category": "onboarding",
        "difficulty": "easy",
    },
    {
        "question": "What happens if I fail the probation review?",
        "category": "onboarding",
        "difficulty": "medium",
    },
    {
        "question": "Who is my point of contact during onboarding?",
        "category": "onboarding",
        "difficulty": "medium",
    },
    # --- Cross-policy / hard ---
    {
        "question": "I am relocating abroad — what policies affect me?",
        "category": "cross_policy",
        "difficulty": "hard",
    },
    {
        "question": "How do vacation and remote-work policies interact?",
        "category": "cross_policy",
        "difficulty": "hard",
    },
    {
        "question": "What benefits am I entitled to during probation?",
        "category": "cross_policy",
        "difficulty": "hard",
    },
    {
        "question": "If I receive a gift from a vendor during a review period, what should I do?",
        "category": "cross_policy",
        "difficulty": "hard",
    },
]
