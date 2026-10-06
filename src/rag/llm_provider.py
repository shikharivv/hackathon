from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from loguru import logger


class FakeLLM(BaseLLM):
    """Deterministic stand-in used when no real LLM backend is configured.

    Returns a fixed message advising the user to set an API key.
    """

    fallback_message: str = (
        "I don't have access to an LLM. "
        "Please configure OPENAI_API_KEY or another supported provider."
    )

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "fake-llm"

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(
            generations=[
                [Generation(text=self.fallback_message)] for _ in prompts
            ]
        )


class LLMProvider:
    """Factory that returns a LangChain-compatible LLM for the requested backend."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        temperature: float = 0.1,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self) -> BaseLLM:
        """Instantiate and return the configured LLM.

        Supported providers:
        * ``openai``       -- requires *api_key* and ``langchain_openai``
        * ``huggingface``  -- local HuggingFace pipeline
        * ``fake``         -- deterministic fallback (no network)

        Raises :class:`ValueError` for unknown providers.
        """

        if self.provider == "openai":
            return self._build_openai()
        if self.provider in {"huggingface", "hf"}:
            return self._build_huggingface()
        if self.provider == "fake":
            return FakeLLM()

        raise ValueError(
            f"Unknown LLM provider '{self.provider}'. "
            "Supported: openai, huggingface, fake."
        )

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_openai(self) -> BaseLLM:
        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided; falling back to FakeLLM"
            )
            return FakeLLM()

        try:
            from langchain_openai import ChatOpenAI  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for the OpenAI provider. "
                "Install it with: pip install langchain-openai"
            ) from exc

        logger.info("Initialising ChatOpenAI (model={})", self.model)
        return ChatOpenAI(  # type: ignore[return-value]
            model=self.model,
            api_key=self.api_key,  # type: ignore[arg-type]
            temperature=self.temperature,
        )

    def _build_huggingface(self) -> BaseLLM:
        try:
            from langchain_community.llms.huggingface_pipeline import (  # noqa: WPS433
                HuggingFacePipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-community and transformers are required for the "
                "HuggingFace provider. Install them with: "
                "pip install langchain-community transformers torch"
            ) from exc

        logger.info(
            "Initialising HuggingFacePipeline (model={})", self.model
        )
        return HuggingFacePipeline.from_model_id(
            model_id=self.model,
            task="text-generation",
            pipeline_kwargs={
                "temperature": self.temperature,
                "max_new_tokens": 1024,
            },
        )
