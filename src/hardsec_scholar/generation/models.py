"""Typed inputs and outputs for grounded answer generation."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hardsec_scholar.domain import Citation, Evidence


class AnswerStatus(str, Enum):
    """Describe whether the local corpus supported an answer."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"


class ClaimDraft(BaseModel):
    """Represent one model-generated claim and its evidence bindings."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class AnswerDraft(BaseModel):
    """Represent constrained model output before citation verification."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    claims: list[ClaimDraft] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> "AnswerDraft":
        """Require claims only for answerable drafts and a gap for refusals."""
        if self.answerable and not self.claims:
            raise ValueError("An answerable draft must contain at least one claim")
        if not self.answerable and self.claims:
            raise ValueError("An unanswerable draft must not contain claims")
        if not self.answerable and not self.missing_evidence:
            raise ValueError("An unanswerable draft must describe missing evidence")
        return self


class VerificationResult(BaseModel):
    """Record deterministic citation integrity checks."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """Return a user-facing answer together with auditable source evidence."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    verification_errors: list[str] = Field(default_factory=list)
    searched_paper_ids: list[str] = Field(default_factory=list)
