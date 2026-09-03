"""Typed planning, grading, verification, trace, and run models."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hardsec_scholar.domain import Evidence, QuestionType
from hardsec_scholar.generation import AnswerDraft, GroundedAnswer


class QuestionPlan(BaseModel):
    """Describe how a paper question should be retrieved and answered."""

    model_config = ConfigDict(extra="forbid")

    question_type: QuestionType
    sub_questions: list[str] = Field(default_factory=list)
    preferred_sections: list[str] = Field(default_factory=list)
    requires_comparison: bool = False


class EvidenceGrade(BaseModel):
    """Describe whether retrieved evidence covers the question."""

    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    selected_evidence_ids: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grade(self) -> "EvidenceGrade":
        """Require selected evidence for success and gaps for failure."""
        if self.sufficient and not self.selected_evidence_ids:
            raise ValueError("A sufficient grade must select evidence")
        if not self.sufficient and not self.missing_aspects:
            raise ValueError("An insufficient grade must describe missing aspects")
        return self


class QueryRewrite(BaseModel):
    """Represent one evidence-gap-driven retrieval query rewrite."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SemanticCitationCheck(BaseModel):
    """Judge whether each generated claim is supported by cited evidence."""

    model_config = ConfigDict(extra="forbid")

    supported: bool
    unsupported_claim_indices: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_check(self) -> "SemanticCitationCheck":
        """Require an explanation whenever semantic verification fails."""
        if self.supported and (self.unsupported_claim_indices or self.errors):
            raise ValueError("A supported answer must not contain verification errors")
        if not self.supported and not self.errors:
            raise ValueError("An unsupported answer must describe verification errors")
        return self


class TraceEvent(BaseModel):
    """Store one safe, user-displayable workflow transition summary."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(gt=0)
    event: str = Field(min_length=1)
    node: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class AgentRun(BaseModel):
    """Return the final answer together with bounded execution metadata."""

    model_config = ConfigDict(extra="forbid")

    answer: GroundedAnswer
    question_type: QuestionType
    plan: QuestionPlan
    search_queries: list[str]
    rewrite_reasons: list[str]
    retrieval_retries: int = Field(ge=0)
    answer_repairs: int = Field(ge=0)
    trace_events: list[TraceEvent]


class ResearchState(TypedDict, total=False):
    """Represent state shared by all LangGraph workflow nodes."""

    question: str
    selected_paper_ids: list[str]
    plan: QuestionPlan
    search_queries: list[str]
    rewrite_reasons: list[str]
    round_queries: list[str]
    retrieved_evidence: list[Evidence]
    selected_evidence: list[Evidence]
    grade: EvidenceGrade
    missing_aspects: list[str]
    retry_count: int
    repair_count: int
    rewrite_valid: bool
    answer_draft: AnswerDraft
    citation_valid: bool
    verification_errors: list[str]
    result: GroundedAnswer
    trace_events: list[TraceEvent]
