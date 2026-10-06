from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: str = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    hf_model_id: str = Field(default="microsoft/Phi-3-mini-4k-instruct")
    hf_api_token: str = Field(default="")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384)


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VECTORSTORE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    path: Path = Field(default=PROJECT_ROOT / "data" / "vectorstore")
    collection: str = Field(default="hr_policies")
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=64, ge=0, le=500)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=3000, ge=500, le=8000)


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="HR Policy RAG Assistant")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
