from __future__ import annotations

import re
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger


class PolicyDocumentLoader:
    """Load HR policy Markdown files and split them into retrieval-ready chunks."""

    _META_PATTERNS: dict[str, re.Pattern[str]] = {
        "policy_number": re.compile(
            r"(?:policy[\s_-]*(?:number|no|#))\s*[:=]\s*(.+)",
            re.IGNORECASE,
        ),
        "title": re.compile(r"^#\s+(.+)", re.MULTILINE),
        "effective_date": re.compile(
            r"(?:effective[\s_-]*date)\s*[:=]\s*(.+)", re.IGNORECASE
        ),
        "version": re.compile(r"(?:version)\s*[:=]\s*(.+)", re.IGNORECASE),
    }

    def __init__(
        self,
        policies_dir: Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.policies_dir = Path(policies_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self) -> list[Document]:
        """Read every ``.md`` file under *policies_dir* and return
        :class:`Document` objects with parsed metadata."""

        if not self.policies_dir.exists():
            logger.warning(
                "Policies directory does not exist: {}", self.policies_dir
            )
            return []

        md_files = sorted(self.policies_dir.rglob("*.md"))
        if not md_files:
            logger.warning(
                "No .md files found in {}", self.policies_dir
            )
            return []

        documents: list[Document] = []
        for filepath in md_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                metadata = self._parse_metadata(content, filepath)
                documents.append(
                    Document(page_content=content, metadata=metadata)
                )
                logger.debug(
                    "Loaded policy document: {} ({})",
                    filepath.name,
                    metadata.get("policy_number", "unknown"),
                )
            except Exception:
                logger.exception("Failed to load {}", filepath)

        logger.info(
            "Loaded {} policy documents from {}", len(documents), self.policies_dir
        )
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split *documents* into smaller chunks while preserving metadata."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            keep_separator=True,
        )

        chunks: list[Document] = []
        for doc in documents:
            splits = splitter.split_documents([doc])
            for idx, chunk in enumerate(splits):
                chunk.metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "source_file": doc.metadata.get("source_file", ""),
                }
                chunks.append(chunk)

        logger.info(
            "Split {} documents into {} chunks (size={}, overlap={})",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def load_and_chunk(self) -> list[Document]:
        """Convenience wrapper: load then chunk all policy documents."""

        documents = self.load_documents()
        if not documents:
            return []
        return self.chunk_documents(documents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_metadata(
        self, content: str, filepath: Path
    ) -> dict[str, str | int]:
        """Extract structured metadata from Markdown front-matter or body."""

        metadata: dict[str, str | int] = {
            "source_file": filepath.name,
            "source_path": str(filepath),
        }

        for key, pattern in self._META_PATTERNS.items():
            match = pattern.search(content)
            if match:
                metadata[key] = match.group(1).strip()

        # Fallback: derive title from filename when not found in content.
        if "title" not in metadata:
            metadata["title"] = filepath.stem.replace("_", " ").replace(
                "-", " "
            ).title()

        return metadata
