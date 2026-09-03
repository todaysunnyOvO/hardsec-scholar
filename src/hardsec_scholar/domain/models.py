"""Typed domain entities used across ingestion, retrieval, and generation."""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchArea(str, Enum):
    """Classify the supported hardware-security research areas."""

    SIDE_CHANNEL_ATTACK = "side_channel_attack"
    ARCHITECTURAL_SECURITY = "architectural_security"
    HARDWARE_FUZZING = "hardware_fuzzing"
    OTHER = "other"


class QuestionType(str, Enum):
    """Classify questions for retrieval planning and answer formatting."""

    FACT = "fact"
    MECHANISM = "mechanism"
    THREAT_MODEL = "threat_model"
    EXPERIMENT = "experiment"
    METRIC = "metric"
    COMPARISON = "comparison"
    LIMITATION = "limitation"


class PaperStatus(str, Enum):
    """Track a paper through ingestion and retrieval indexing."""

    PENDING = "pending"
    PARSED = "parsed"
    INDEXED = "indexed"
    FAILED = "failed"


class PaperMetadata(BaseModel):
    """Store normalized metadata for an indexed paper."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    research_area: ResearchArea = ResearchArea.OTHER
    file_path: Path
    page_count: int = Field(gt=0)
    status: PaperStatus = PaperStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaperChunk(BaseModel):
    """Represent a page-aware section fragment from a paper."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str | None = None
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    source_type: str = "paragraph"
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "PaperChunk":
        """Require page ranges to be ordered."""
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be smaller than page_start")
        return self


class Evidence(BaseModel):
    """Represent a retrievable and rankable paper fragment."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    section: str | None = None
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    dense_rank: int | None = Field(default=None, gt=0)
    dense_score: float | None = None
    bm25_rank: int | None = Field(default=None, gt=0)
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    @model_validator(mode="after")
    def validate_page_range(self) -> "Evidence":
        """Require page ranges to be ordered."""
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be smaller than page_start")
        return self


class Citation(BaseModel):
    """Bind a generated claim to one selected evidence fragment."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    section: str | None = None
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    claim: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> "Citation":
        """Require page ranges to be ordered."""
        if self.page_end < self.page_start:
            raise ValueError("page_end must not be smaller than page_start")
        return self


class HardSecEvidence(BaseModel):
    """Describe domain-specific facts extracted from cited evidence."""

    model_config = ConfigDict(extra="forbid")

    research_area: ResearchArea | None = None
    attack_or_defense: str | None = None
    target: list[str] = Field(default_factory=list)
    threat_model: str | None = None
    attacker_capabilities: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    platform: list[str] = Field(default_factory=list)
    evaluation_type: str | None = None
    metrics: dict[str, str] = Field(default_factory=dict)
    overhead: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
