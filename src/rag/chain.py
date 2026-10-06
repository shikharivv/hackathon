from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from loguru import logger

from src.rag.guardrails import GuardrailResult, ResponseGuardrails

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class Citation:
    """A single reference to a source policy chunk."""

    policy_number: str
    policy_title: str
    section: str
    chunk_text: str
    relevance_score: float


@dataclass
class RAGResponse:
    """Structured response returned by :meth:`HRPolicyRAGChain.query`."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    sources_used: int = 0
    fallback_triggered: bool = False
    guardrail_flags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


# ------------------------------------------------------------------
# RAG chain
# ------------------------------------------------------------------

class HRPolicyRAGChain:
    """End-to-end Retrieval-Augmented Generation chain for HR policy Q&A.

    The chain enforces strict source-grounding: if the retrieved context is
    insufficient the model is instructed (and programmatically forced) to
    decline rather than hallucinate.
    """

    SYSTEM_PROMPT: str = (
        "You are an HR Policy Assistant.  Your ONLY purpose is to answer "
        "questions about company HR policies based on the context provided "
        "below.\n\n"
        "RULES YOU MUST FOLLOW:\n"
        "1. Answer ONLY based on the provided context. Never use outside "
        "knowledge.\n"
        "2. Always cite the source document by its policy number and title "
        '   (e.g., "[Policy HR-001: Leave Policy]").\n'
        '3. If the context does not contain enough information, say: "I don\'t '
        "have enough information in the available HR policies to answer that "
        'question."\n'
        "4. Never fabricate, guess, or assume policy details that are not "
        "explicitly stated in the context.\n"
        "5. Do not provide legal advice or medical advice.\n"
        "6. Be concise and professional.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseLLM,
        confidence_threshold: float = 0.65,
        max_context_tokens: int = 3000,
        top_k: int = 5,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self._guardrails = ResponseGuardrails(
            confidence_threshold=confidence_threshold
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for *question*.

        Steps:
        1. Retrieve top-k documents with similarity scores.
        2. Filter by confidence threshold.
        3. If no documents survive filtering, return a fallback response.
        4. Build context string from surviving chunks.
        5. Invoke the LLM.
        6. Extract citations and run guardrails.
        7. Return a structured :class:`RAGResponse`.
        """

        start = time.perf_counter()

        # 1. Retrieve
        results_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=self.top_k
        )

        # ChromaDB returns *distance* (lower = better).  Convert to a 0-1
        # similarity score so downstream logic stays consistent.
        scored: list[tuple[Document, float]] = []
        for doc, distance in results_with_scores:
            similarity = max(0.0, 1.0 - distance)
            scored.append((doc, similarity))

        # 2. Filter
        filtered = [
            (doc, score)
            for doc, score in scored
            if score >= self.confidence_threshold
        ]

        # 3. Fallback
        if not filtered:
            latency = (time.perf_counter() - start) * 1000
            logger.info(
                "No documents above threshold ({}) for query: {}",
                self.confidence_threshold,
                question[:80],
            )
            return RAGResponse(
                answer=(
                    "I don't have enough information in the available HR "
                    "policies to answer that question."
                ),
                confidence=0.0,
                sources_used=0,
                fallback_triggered=True,
                guardrail_flags=["no_sources_found"],
                latency_ms=round(latency, 2),
            )

        docs = [doc for doc, _ in filtered]
        scores = [score for _, score in filtered]

        # 4. Build context
        context = self._build_context(docs)

        # 5. Invoke LLM
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)
        try:
            llm_output = self.llm.invoke(prompt)
            # ChatModels return an AIMessage; plain LLMs return a string.
            answer = str(llm_output.content) if hasattr(llm_output, "content") else str(llm_output)
        except Exception:
            logger.exception("LLM invocation failed")
            answer = (
                "I encountered an error while processing your question. "
                "Please try again later."
            )

        # 6. Citations & guardrails
        citations = self._extract_citations(answer, filtered)
        source_texts = [doc.page_content for doc in docs]
        guardrail = self._check_guardrails(answer, source_texts, scores)

        # If guardrails fail, replace answer with safe fallback
        if not guardrail.passed:
            logger.warning(
                "Guardrails triggered ({}); overriding response",
                guardrail.flags,
            )
            answer = (
                "I don't have enough information in the available HR "
                "policies to confidently answer that question. "
                "Please consult your HR department directly."
            )

        latency = (time.perf_counter() - start) * 1000
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=round(avg_confidence, 4),
            sources_used=len(docs),
            fallback_triggered=not guardrail.passed,
            guardrail_flags=guardrail.flags,
            latency_ms=round(latency, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, documents: list[Document]) -> str:
        """Concatenate document chunks into a single context string,
        respecting *max_context_tokens* (approximated as characters / 4)."""

        max_chars = self.max_context_tokens * 4  # rough token estimate
        parts: list[str] = []
        total = 0

        for doc in documents:
            meta = doc.metadata
            header = (
                f"[Source: {meta.get('policy_number', 'N/A')} - "
                f"{meta.get('title', 'Untitled')}]"
            )
            block = f"{header}\n{doc.page_content}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)

        return "\n---\n".join(parts)

    def _extract_citations(
        self,
        response: str,
        scored_docs: list[tuple[Document, float]],
    ) -> list[Citation]:
        """Build :class:`Citation` objects from the scored source documents."""

        citations: list[Citation] = []
        for doc, score in scored_docs:
            meta = doc.metadata
            citations.append(
                Citation(
                    policy_number=meta.get("policy_number", "N/A"),
                    policy_title=meta.get("title", "Untitled"),
                    section=meta.get("source_file", ""),
                    chunk_text=doc.page_content[:200],
                    relevance_score=round(score, 4),
                )
            )
        return citations

    def _check_guardrails(
        self,
        response: str,
        source_texts: list[Document] | list[str],
        scores: list[float],
    ) -> GuardrailResult:
        """Delegate to :class:`ResponseGuardrails` for all safety checks."""

        texts = [
            t if isinstance(t, str) else t.page_content for t in source_texts
        ]
        return self._guardrails.apply_all(response, texts, scores)
