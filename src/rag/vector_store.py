from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger


class VectorStoreManager:
    """Manage a ChromaDB-backed vector store for HR policy documents."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._store: Chroma | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_store(self, documents: list[Document]) -> Chroma:
        """Create a **new** vector store from *documents* and persist it."""

        if not documents:
            raise ValueError("Cannot create a vector store with zero documents")

        logger.info(
            "Creating vector store with {} documents at {}",
            len(documents),
            self.persist_directory,
        )

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
        )

        logger.info(
            "Vector store created: collection='{}', docs={}",
            self.collection_name,
            len(documents),
        )
        return self._store

    def load_store(self) -> Chroma:
        """Load an existing persisted vector store from disk."""

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"Vector store directory not found: {self.persist_directory}"
            )

        logger.info(
            "Loading vector store from {}", self.persist_directory
        )

        self._store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
        )

        logger.info("Vector store loaded successfully")
        return self._store

    def get_or_create(
        self, documents: list[Document] | None = None
    ) -> Chroma:
        """Load the vector store if it already exists, otherwise create it.

        When the store does not exist, *documents* **must** be provided.
        """

        if self.persist_directory.exists() and any(
            self.persist_directory.iterdir()
        ):
            logger.debug("Existing vector store found; loading")
            return self.load_store()

        if documents is None:
            raise ValueError(
                "No existing vector store found and no documents provided to "
                "create one"
            )

        return self.create_store(documents)

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[Document]:
        """Return the *k* most similar documents for *query*."""

        store = self._ensure_store()
        results = store.similarity_search(query, k=k)
        logger.debug("Similarity search returned {} results for query", len(results))
        return results

    def similarity_search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return the *k* most similar documents with their distance scores."""

        store = self._ensure_store()
        results = store.similarity_search_with_score(query, k=k)
        logger.debug(
            "Similarity search (with scores) returned {} results", len(results)
        )
        return results

    def delete_store(self) -> None:
        """Remove all persisted vector store data from disk."""

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            logger.info("Deleted vector store at {}", self.persist_directory)
        self._store = None

    def get_collection_stats(self) -> dict[str, object]:
        """Return basic statistics about the underlying collection."""

        store = self._ensure_store()
        collection = store._collection  # noqa: SLF001
        count = collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_store(self) -> Chroma:
        """Return the cached store or attempt to load it."""

        if self._store is not None:
            return self._store
        return self.load_store()
