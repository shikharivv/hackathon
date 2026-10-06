"""Tests for src.config.settings — application configuration models."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# LLMSettings
# ---------------------------------------------------------------------------

def test_llm_settings_defaults():
    """LLMSettings should expose sensible defaults when no env vars are set."""
    from src.config.settings import LLMSettings

    settings = LLMSettings()

    assert settings.llm_provider == "openai"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.temperature == 0.1
    assert settings.hf_model_id == "microsoft/Phi-3-mini-4k-instruct"


# ---------------------------------------------------------------------------
# EmbeddingSettings
# ---------------------------------------------------------------------------

def test_embedding_settings_defaults():
    """EmbeddingSettings should default to MiniLM with 384 dimensions."""
    from src.config.settings import EmbeddingSettings

    settings = EmbeddingSettings()

    assert settings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.dimension == 384


# ---------------------------------------------------------------------------
# VectorStoreSettings
# ---------------------------------------------------------------------------

def test_vectorstore_settings_defaults():
    """VectorStoreSettings should point to <project>/data/vectorstore."""
    from src.config.settings import VectorStoreSettings

    settings = VectorStoreSettings()

    assert settings.collection == "hr_policies"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k == 5
    assert isinstance(settings.path, Path)


# ---------------------------------------------------------------------------
# AppSettings / get_settings
# ---------------------------------------------------------------------------

def test_app_settings_contains_nested():
    """AppSettings should compose all sub-settings as nested attributes."""
    from src.config.settings import AppSettings

    settings = AppSettings()

    assert settings.app_name == "HR Policy RAG Assistant"
    assert settings.version == "1.0.0"
    assert hasattr(settings, "llm")
    assert hasattr(settings, "embedding")
    assert hasattr(settings, "vectorstore")
    assert hasattr(settings, "rag")
    assert hasattr(settings, "api")


def test_get_settings_singleton():
    """get_settings should return the same cached instance on repeated calls."""
    from src.config.settings import get_settings

    first = get_settings()
    second = get_settings()

    assert first is second
