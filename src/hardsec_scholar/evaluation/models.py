"""Typed records for the hardware-security offline evaluation set."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationCategory(str, Enum):
    """Group benchmark questions by the capability they exercise."""

    SINGLE_PAPER_FACT = "single_paper_fact"
    MECHANISM = "mechanism"
    THREAT_MODEL = "threat_model"
    EXPERIMENT = "experiment"
    COMPARISON = "comparison"
    UNANSWERABLE = "unanswerable"


class EvaluationSample(BaseModel):
    """Represent one human-reviewable, evidence-grounded evaluation question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^hardsec_eval_\d{3}$")
    question: str = Field(min_length=5)
    category: EvaluationCategory
    paper_ids: list[str] = Field(default_factory=list)
    reference_answer: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    should_abstain: bool = False

    @model_validator(mode="after")
    def validate_ground_truth_shape(self) -> "EvaluationSample":
        """Keep answerable and abstention records internally consistent."""
        is_unanswerable = self.category is EvaluationCategory.UNANSWERABLE
        if self.should_abstain != is_unanswerable:
            raise ValueError(
                "should_abstain must be true exactly for unanswerable samples"
            )
        if self.should_abstain:
            if self.paper_ids or self.relevant_chunk_ids or self.expected_pages:
                raise ValueError(
                    "unanswerable samples cannot declare gold papers, chunks, or pages"
                )
            return self

        if not self.paper_ids or not self.relevant_chunk_ids or not self.expected_pages:
            raise ValueError(
                "answerable samples require gold papers, chunks, and pages"
            )
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("paper_ids must be unique")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must be unique")
        if sorted(set(self.expected_pages)) != self.expected_pages:
            raise ValueError("expected_pages must be sorted and unique")
        if any(page <= 0 for page in self.expected_pages):
            raise ValueError("expected_pages must contain positive page numbers")
        if self.category is EvaluationCategory.COMPARISON and len(self.paper_ids) < 2:
            raise ValueError("comparison samples require at least two papers")
        if (
            self.category is EvaluationCategory.SINGLE_PAPER_FACT
            and len(self.paper_ids) != 1
        ):
            raise ValueError("single-paper facts require exactly one paper")
        return self


class EvaluationVariant(str, Enum):
    """Name the two systems compared by the benchmark."""

    BASELINE = "baseline_dense_rag"
    AGENTIC = "agentic_hybrid_rag"


class RetrievalMetrics(BaseModel):
    """Store deterministic initial-ranking and all-round retrieval metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_5: float | None = Field(default=None, ge=0, le=1)
    recall_at_10: float | None = Field(default=None, ge=0, le=1)
    mrr: float | None = Field(default=None, ge=0, le=1)
    first_relevant_rank: int | None = Field(default=None, gt=0)
    paper_hit: bool | None = None
    page_hit: bool | None = None
    all_round_recall: float | None = Field(default=None, ge=0, le=1)
    retrieved_chunk_count: int = Field(default=0, ge=0)


class AnswerMetrics(BaseModel):
    """Store refusal, lexical answer, and exact-gold citation metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    abstention_correct: bool
    reference_token_f1: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)
    citation_page_precision: float | None = Field(default=None, ge=0, le=1)
    citation_paper_precision: float | None = Field(default=None, ge=0, le=1)
    grounded: bool
    citation_count: int = Field(default=0, ge=0)


class TokenUsage(BaseModel):
    """Record model tokens and a conservative cache-miss cost estimate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_usd: float = Field(default=0.0, ge=0)


class EvaluationRunResult(BaseModel):
    """Persist one sample-system run so interrupted evaluations can resume."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    variant: EvaluationVariant
    success: bool
    error: str | None = None
    latency_seconds: float = Field(ge=0)
    answer_status: str | None = None
    answer: str = ""
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    retrieval: RetrievalMetrics | None = None
    answer_metrics: AnswerMetrics | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    retrieval_retries: int = Field(default=0, ge=0)
    answer_repairs: int = Field(default=0, ge=0)
    query_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
