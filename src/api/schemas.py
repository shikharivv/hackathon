from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Inbound payload for the RAG query endpoint."""

    question: str = Field(..., min_length=5, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)


class CitationSchema(BaseModel):
    """A single citation referencing a policy section."""

    policy_number: str
    policy_title: str
    section: str
    relevance_score: float
    chunk_text: str = ""
    version: str = ""


class QueryResponse(BaseModel):
    """Structured response returned by the RAG pipeline."""

    answer: str
    citations: list[CitationSchema]
    confidence: float
    sources_used: int
    fallback_triggered: bool
    guardrail_flags: list[str]
    latency_ms: float
    answer_mode: str = "generated"
    suggested_questions: list[str] = Field(default_factory=list)
    escalation: dict | None = None
    show_confidence: bool = True


class HealthResponse(BaseModel):
    """Application health-check payload."""

    status: str
    version: str
    vectorstore_loaded: bool
    llm_available: bool
    documents_indexed: int


class IndexResponse(BaseModel):
    """Result of a document (re-)indexing operation."""

    status: str
    documents_loaded: int
    chunks_created: int
    collection_name: str


class FeedbackRequest(BaseModel):
    """User feedback on a specific answer."""

    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(default="")
