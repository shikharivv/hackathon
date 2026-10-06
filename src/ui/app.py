from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from loguru import logger

from src.config.settings import get_settings

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

_EXAMPLE_QUESTIONS: list[str] = [
    "How many vacation days do I get after 3 years?",
    "What is the 401(k) matching policy?",
    "Can I work remotely from another country?",
    "What happens during a Performance Improvement Plan?",
    "What is the gift policy limit?",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _api_available() -> bool:
    """Return *True* if the FastAPI backend is reachable."""
    try:
        resp = httpx.get(f"{_API_BASE}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _query_api(question: str, top_k: int) -> dict[str, Any]:
    """Send a query to the FastAPI backend and return the parsed response."""
    resp = httpx.post(
        f"{_API_BASE}/query",
        json={"question": question, "top_k": top_k, "include_sources": True, "confidence_threshold": st.session_state.confidence_threshold},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _query_direct(question: str, top_k: int) -> dict[str, Any]:
    """Fall back to invoking the RAG chain directly (no API server)."""
    return _runtime().query(question, top_k, st.session_state.confidence_threshold)


@st.cache_resource
def _runtime():
    from src.rag.runtime import PolicyRuntime
    return PolicyRuntime()


def _confidence_badge(confidence: float) -> str:
    """Return a coloured Markdown badge for the confidence score."""
    if confidence >= 0.80:
        return f":green[Evidence relevance: {confidence:.2f}]"
    if confidence >= 0.50:
        return f":orange[Evidence relevance: {confidence:.2f}]"
    return f":red[Evidence relevance: {confidence:.2f}]"


def _list_policies_on_disk() -> list[str]:
    """Return filenames of policy Markdown files found in docs/policies/."""
    settings = get_settings()
    policies_dir = Path(settings.vectorstore.path).parent.parent / "docs" / "policies"
    if not policies_dir.exists():
        return []
    return sorted(p.name for p in policies_dir.glob("*.md"))


def _reindex() -> str:
    """Trigger document re-indexing via the API."""
    if os.getenv('APP_MODE', 'direct') == 'direct':
        documents, sections = _runtime().reindex()
        return f'Indexed {documents} policies ({sections} sections).'
    try:
        resp = httpx.post(f"{_API_BASE}/index", timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Indexed **{data['documents_loaded']}** documents "
            f"(**{data['chunks_created']}** chunks) into "
            f"`{data['collection_name']}`."
        )
    except Exception as exc:
        return f"Re-indexing failed: {exc}"


# ------------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------------

def main() -> None:
    """Entry-point for the Streamlit HR Policy Assistant dashboard."""

    st.set_page_config(
        page_title="HR Policy Assistant",
        page_icon="💼",
        layout="wide",
    )

    # -- Session state initialisation --------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))

    # -- Sidebar -----------------------------------------------------
    with st.sidebar:
        st.header("HR Policy Assistant")

        st.subheader("About")
        st.markdown(
            "This assistant answers HR-policy questions using "
            "**Retrieval-Augmented Generation (RAG)**. It retrieves "
            "relevant policy sections from the knowledge base and uses "
            "an LLM to compose a grounded answer with citations."
        )

        st.subheader("Policies on file")
        policies = _list_policies_on_disk()
        if policies:
            for name in policies:
                st.text(f"  {name}")
        else:
            st.info("No policy documents found.")

        st.subheader("Settings")
        st.session_state.top_k = st.slider(
            "Top-K results", min_value=1, max_value=20, value=st.session_state.top_k
        )
        st.session_state.confidence_threshold = st.slider(
            "Evidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
        )

        if st.button("Re-index Documents"):
            with st.spinner("Indexing..."):
                result_msg = _reindex()
            st.success(result_msg)

    # -- Main area ---------------------------------------------------
    st.title("HR Policy Assistant")
    st.caption("Fictional demo policies. Evidence scores measure retrieval relevance, not answer accuracy.")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("extra"):
                _render_extra(msg["extra"])

    # Example question buttons (shown only when history is empty)
    if not st.session_state.messages:
        st.markdown("**Try an example question:**")
        cols = st.columns(len(_EXAMPLE_QUESTIONS))
        for col, eq in zip(cols, _EXAMPLE_QUESTIONS, strict=True):
            if col.button(eq, key=f"ex_{eq[:20]}"):
                _handle_question(eq)
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask an HR policy question..."):
        _handle_question(prompt)
        st.rerun()


# ------------------------------------------------------------------
# Chat logic
# ------------------------------------------------------------------

def _handle_question(question: str) -> None:
    """Process *question*: call backend, store messages, and display."""

    st.session_state.messages.append({"role": "user", "content": question})

    use_api = os.getenv("APP_MODE", "direct") == "api" and _api_available()
    try:
        if use_api:
            data = _query_api(question, st.session_state.top_k)
        else:
            data = _query_direct(question, st.session_state.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        data = {
            "answer": f"An error occurred: {exc}",
            "citations": [],
            "confidence": 0.0,
            "sources_used": 0,
            "fallback_triggered": True,
            "guardrail_flags": ["error"],
            "latency_ms": 0.0,
        }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "extra": data,
        }
    )


def _render_extra(data: dict[str, Any]) -> None:
    """Render citations, confidence badge, and fallback notice."""

    # Confidence badge
    confidence = data.get("confidence", 0.0)
    st.markdown(_confidence_badge(confidence))

    # Fallback notice
    if data.get("fallback_triggered"):
        st.info(
            "The assistant could not find sufficiently relevant information "
            "in the HR policy knowledge base. The answer above may be "
            "incomplete or a general response."
        )

    # Citations
    citations = data.get("citations", [])
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for cit in citations:
                st.markdown(
                    f"- **{cit.get('policy_number', 'N/A')}** | "
                    f"{cit.get('section', 'N/A')} | "
                    f"relevance: {cit.get('relevance_score', 0.0):.2f}"
                )

                if cit.get('chunk_text'):
                    st.markdown(cit['chunk_text'])

    # Latency
    latency = data.get("latency_ms", 0.0)
    if latency:
        st.caption(f"Latency: {latency:.0f} ms")


# ------------------------------------------------------------------
# Script entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
