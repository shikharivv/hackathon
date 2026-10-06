"""Tests for src.rag.llm_provider — LLM backend factory."""

from __future__ import annotations

import pytest

try:
    from src.rag.llm_provider import FakeLLM, LLMProvider

    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="langchain/torch not available")


def test_fake_llm_type():
    """FakeLLM should identify itself as 'fake-llm'."""
    llm = FakeLLM()
    assert llm._llm_type == "fake-llm"


def test_fake_llm_generates_fallback():
    """FakeLLM should return its fallback message for any prompt."""
    llm = FakeLLM()
    result = llm.invoke("What is the leave policy?")
    assert "don't have access" in str(result).lower() or "api key" in str(result).lower()


def test_provider_fake_returns_fake_llm():
    """Requesting provider='fake' should return a FakeLLM instance."""
    provider = LLMProvider(provider="fake")
    llm = provider.get_llm()
    assert isinstance(llm, FakeLLM)


def test_provider_openai_without_key_returns_fake():
    """OpenAI provider with no API key should fall back to FakeLLM."""
    provider = LLMProvider(provider="openai", api_key="")
    llm = provider.get_llm()
    assert isinstance(llm, FakeLLM)


def test_provider_unknown_raises():
    """An unsupported provider name should raise ValueError."""
    provider = LLMProvider(provider="unknown_backend")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        provider.get_llm()
