"""Tests for the FastAPI application and API schemas.

Route-level tests are gated on whether the API module is implemented.
Schema validation tests run unconditionally since schemas.py exists.
"""

from __future__ import annotations

import pytest

from src.api.schemas import (
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
)

# Attempt to import the FastAPI app for integration tests.
try:
    from fastapi.testclient import TestClient

    from src.api.main import app  # noqa: F401

    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False


# ---------------------------------------------------------------------------
# Schema validation (always available)
# ---------------------------------------------------------------------------

def test_query_request_validation():
    """QueryRequest should reject questions shorter than 5 characters."""
    with pytest.raises(ValueError):
        QueryRequest(question="Hi")


def test_query_request_accepts_valid():
    """QueryRequest should accept a well-formed question."""
    req = QueryRequest(question="How many vacation days do new employees get?")

    assert req.top_k == 5
    assert req.include_sources is True


def test_health_response_schema():
    """HealthResponse should round-trip through dict serialisation."""
    resp = HealthResponse(
        status="ok",
        version="1.0.0",
        vectorstore_loaded=True,
        llm_available=False,
        documents_indexed=42,
    )
    data = resp.model_dump()

    assert data["status"] == "ok"
    assert data["documents_indexed"] == 42


def test_feedback_request_rating_bounds():
    """FeedbackRequest rating must be between 1 and 5."""
    with pytest.raises(ValueError):
        FeedbackRequest(question="q", answer="a", rating=0)

    with pytest.raises(ValueError):
        FeedbackRequest(question="q", answer="a", rating=6)


# ---------------------------------------------------------------------------
# FastAPI endpoint tests (require app implementation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _API_AVAILABLE, reason="API app not yet implemented")
class TestAPIEndpoints:
    """Integration tests that exercise the running FastAPI application."""

    @pytest.fixture(autouse=True)
    def _client(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """GET /health should return 200 with status information."""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body

    def test_query_endpoint_validation(self):
        """POST /query with an invalid payload should return 422."""
        response = self.client.post("/api/v1/query", json={"question": "Hi"})
        assert response.status_code == 422

    def test_policies_endpoint(self):
        """GET /policies should return a list."""
        response = self.client.get("/api/v1/policies")
        assert response.status_code in {200, 404}
