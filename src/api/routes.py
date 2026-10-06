from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from loguru import logger

from src.api.schemas import (
    CitationSchema,
    FeedbackRequest,
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from src.config.settings import get_settings

router = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return current application health status."""

    settings = get_settings()
    state = request.app.state

    vectorstore_loaded: bool = getattr(state, "vectorstore", None) is not None
    llm_available: bool = getattr(state, "llm", None) is not None
    documents_indexed: int = getattr(state, "documents_indexed", 0)

    return HealthResponse(
        status="ok" if vectorstore_loaded and llm_available else "degraded",
        version=settings.version,
        vectorstore_loaded=vectorstore_loaded,
        llm_available=llm_available,
        documents_indexed=documents_indexed,
    )


# ------------------------------------------------------------------
# POST /query
# ------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Execute a RAG query against the HR policy knowledge base."""

    start = time.perf_counter()
    rag_chain = getattr(request.app.state, "rag_chain", None)

    if rag_chain is None:
        return QueryResponse(
            answer="The RAG pipeline is not initialised. Please index documents first.",
            citations=[],
            confidence=0.0,
            sources_used=0,
            fallback_triggered=True,
            guardrail_flags=["rag_chain_unavailable"],
            latency_ms=_elapsed_ms(start),
        )

    try:
        result: dict[str, Any] = rag_chain.query(
            question=body.question,
            top_k=body.top_k,
            confidence_threshold=body.confidence_threshold,
        )
    except Exception:
        logger.exception("RAG query failed for question: {}", body.question)
        return QueryResponse(
            answer="An internal error occurred while processing your question.",
            citations=[],
            confidence=0.0,
            sources_used=0,
            fallback_triggered=True,
            guardrail_flags=["internal_error"],
            latency_ms=_elapsed_ms(start),
        )

    citations: list[CitationSchema] = []
    if body.include_sources:
        for src in result.get("citations", []):
            citations.append(
                CitationSchema(
                    policy_number=src.get("policy_number", "N/A"),
                    policy_title=src.get("policy_title", "N/A"),
                    section=src.get("section", "N/A"),
                    chunk_text=src.get("chunk_text", ""),
                    version=src.get("version", ""),
                    relevance_score=round(float(src.get("relevance_score", 0.0)), 4),
                )
            )

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=citations,
        confidence=round(float(result.get("confidence", 0.0)), 4),
        sources_used=len(citations),
        fallback_triggered=result.get("fallback_triggered", False),
        guardrail_flags=result.get("guardrail_flags", []),
        latency_ms=_elapsed_ms(start),
        answer_mode=result.get("answer_mode", "generated"),
        suggested_questions=result.get("suggested_questions", []),
        escalation=result.get("escalation"),
    )


# ------------------------------------------------------------------
# POST /index
# ------------------------------------------------------------------
@router.post("/index", response_model=IndexResponse)
async def index_documents(request: Request) -> IndexResponse:
    """Trigger (re-)indexing of policy documents into the vector store."""

    runtime = request.app.state.rag_chain
    documents, sections = runtime.reindex()
    request.app.state.documents_indexed = documents
    return IndexResponse(status='ok', documents_loaded=documents,
                         chunks_created=sections, collection_name='policy_sections')


# ------------------------------------------------------------------
# POST /feedback
# ------------------------------------------------------------------
@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest) -> dict[str, str]:
    """Record user feedback for a given question-answer pair."""

    logger.info(
        "Feedback received â€” rating={} question='{}'",
        body.rating,
        body.question[:80],
    )
    # In production this would persist to a database or analytics service.
    return {"status": "recorded"}


# ------------------------------------------------------------------
# GET /policies
# ------------------------------------------------------------------
@router.get("/policies")
async def list_policies() -> list[dict[str, str]]:
    """Return metadata for every policy document on disk."""

    from src.rag.runtime import PolicyRuntime
    runtime = PolicyRuntime()
    policies = {}
    for section in runtime.sections:
        policies[section['policy_number']] = dict(policy_number=section['policy_number'],
                                                 title=section['policy_title'], version=section['version'])
    return list(policies.values())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since *start*."""
    return round((time.perf_counter() - start) * 1000, 2)
