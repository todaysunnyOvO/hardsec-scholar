"""Reasoning interfaces used by the bounded Agentic RAG graph."""

from typing import Protocol

from hardsec_scholar.agent.models import (
    EvidenceGrade,
    QueryRewrite,
    QuestionPlan,
    SemanticCitationCheck,
)
from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation import AnswerDraft


class AgentReasoner(Protocol):
    """Make structured planning and verification decisions for graph nodes."""

    def classify_and_plan(self, question: str) -> QuestionPlan:
        """Classify the question and create a bounded retrieval plan."""
        ...

    def grade_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        selected_paper_ids: list[str],
        requires_comparison: bool,
    ) -> EvidenceGrade:
        """Select sufficient evidence or describe concrete gaps."""
        ...

    def rewrite_query(
        self,
        question: str,
        *,
        previous_queries: list[str],
        missing_aspects: list[str],
    ) -> QueryRewrite:
        """Create a new query targeted at the current evidence gap."""
        ...

    def verify_citations(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
    ) -> SemanticCitationCheck:
        """Check that each cited fragment semantically supports its claim."""
        ...

    def repair_answer(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
        *,
        verification_errors: list[str],
    ) -> AnswerDraft:
        """Repair unsupported claims without introducing new evidence IDs."""
        ...
