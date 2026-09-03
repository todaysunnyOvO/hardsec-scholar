"""Typed application configuration for HardSec Scholar."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseModel):
    """Configure candidate retrieval, fusion, and reranking."""

    model_config = ConfigDict(extra="forbid")

    dense_top_k: int = Field(default=20, gt=0)
    bm25_top_k: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=6, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    collection_name: str = Field(default="hardsec_papers", min_length=1)
    reranker_enabled: bool = True
    reranker_model: str = Field(default="ms-marco-TinyBERT-L-2-v2", min_length=1)
    reranker_cache: Path = Path("data/models/flashrank")

    @model_validator(mode="after")
    def validate_rerank_window(self) -> "RetrievalSettings":
        """Ensure reranking cannot request more unique candidates than retrieval."""
        if self.rerank_top_k > self.dense_top_k + self.bm25_top_k:
            raise ValueError(
                "rerank_top_k exceeds the maximum retrieval candidate count"
            )
        return self


class AgentSettings(BaseModel):
    """Configure bounded Agentic RAG loops."""

    model_config = ConfigDict(extra="forbid")

    max_retrieval_retries: int = Field(default=2, ge=0, le=5)
    max_answer_repairs: int = Field(default=1, ge=0, le=3)
    max_queries_per_round: int = Field(default=6, gt=0, le=20)
    web_search_default: bool = False


class GenerationSettings(BaseModel):
    """Configure evidence context and deterministic refusal behavior."""

    model_config = ConfigDict(extra="forbid")

    max_context_evidence: int = Field(default=6, gt=0, le=20)
    max_evidence_per_paper: int = Field(default=3, gt=0, le=10)
    refusal_message: str = Field(
        default=(
            "The indexed papers do not provide enough evidence to answer this question."
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_context_limits(self) -> "GenerationSettings":
        """Keep the per-paper cap within the complete context window."""
        if self.max_evidence_per_paper > self.max_context_evidence:
            raise ValueError(
                "max_evidence_per_paper must not exceed max_context_evidence"
            )
        return self


class ChunkingSettings(BaseModel):
    """Configure section-aware paper chunking."""

    model_config = ConfigDict(extra="forbid")

    chunk_size_tokens: int = Field(default=700, ge=100)
    overlap_tokens: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingSettings":
        """Require overlap to be smaller than the target chunk size."""
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        return self


class IngestionSettings(BaseModel):
    """Configure accepted paper files and ingestion fallbacks."""

    model_config = ConfigDict(extra="forbid")

    max_file_size_mb: int = Field(default=50, gt=0, le=500)
    exclude_references: bool = True
    accepted_extensions: list[str] = Field(default_factory=lambda: [".pdf"])


class PathSettings(BaseModel):
    """Configure local persistent paths."""

    model_config = ConfigDict(extra="forbid")

    papers: Path = Path("data/papers")
    chroma: Path = Path("data/chroma")
    evaluations: Path = Path("data/evaluations")
    database: Path = Path("data/hardsec_scholar.db")


class AppSettings(BaseModel):
    """Represent non-secret application settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class RuntimeSettings(BaseSettings):
    """Load model credentials and runtime switches from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "openai"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    embedding_provider: str = "openai"
    embedding_model: str = ""
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    allow_web_search: bool = False
    web_search_provider: str = "tavily"
    web_search_api_key: SecretStr | None = None


def load_app_settings(path: Path | str = "config/default.yaml") -> AppSettings:
    """Load and validate non-secret application settings from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        raw: Any = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise ValueError("Application configuration must be a YAML mapping")
    return AppSettings.model_validate(raw)
