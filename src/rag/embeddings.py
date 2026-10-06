from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger


class EmbeddingManager:
    """Thin wrapper around a HuggingFace sentence-transformer embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._embedding_fn: HuggingFaceEmbeddings | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embedding_function(self) -> Embeddings:
        """Return a LangChain-compatible :class:`Embeddings` instance.

        The model is loaded lazily on first call and then cached.
        """

        if self._embedding_fn is None:
            logger.info("Loading embedding model: {}", self.model_name)
            self._embedding_fn = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")
        return self._embedding_fn

    def embed_text(self, text: str) -> list[float]:
        """Compute the embedding vector for a single *text* string."""

        fn = self.get_embedding_function()
        return fn.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of *texts*."""

        fn = self.get_embedding_function()
        return fn.embed_documents(texts)
