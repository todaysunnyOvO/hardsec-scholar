"""Orchestrate one-pass grounded retrieval, generation, and verification."""

from hardsec_scholar.config import GenerationSettings
from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation.context import EvidenceContextBuilder
from hardsec_scholar.generation.interfaces import AnswerGenerator, EvidenceRetriever
from hardsec_scholar.generation.models import (
    AnswerDraft,
    AnswerStatus,
    GroundedAnswer,
)
from hardsec_scholar.generation.verifier import CitationVerifier


def render_claims(draft: AnswerDraft) -> str:
    """Render only schema-validated claims with inline evidence IDs."""
    return "\n\n".join(
        f"{claim.text} [{', '.join(claim.evidence_ids)}]" for claim in draft.claims
    )


class BasicRAGService:
    """Answer once from local evidence and fail closed when grounding is invalid."""

    def __init__(
        self,
        *,
        retriever: EvidenceRetriever,
        generator: AnswerGenerator,
        settings: GenerationSettings,
        verifier: CitationVerifier | None = None,
    ) -> None:
        """Store providers and construct a bounded context builder."""
        self.retriever = retriever
        self.generator = generator
        self.settings = settings
        self.verifier = verifier or CitationVerifier()
        self.context_builder = EvidenceContextBuilder(
            max_evidence=settings.max_context_evidence,
            max_per_paper=settings.max_evidence_per_paper,
        )

    def answer(
        self, question: str, *, paper_ids: list[str] | None = None
    ) -> GroundedAnswer:
        """Return a verified grounded answer or an explicit refusal."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        searched_paper_ids = list(paper_ids or [])
        ranked = self.retriever.search(question, paper_ids=paper_ids)
        selected = self.context_builder.select(ranked)
        if not selected:
            return self._abstain(
                missing_evidence=["No relevant evidence was retrieved."],
                searched_paper_ids=searched_paper_ids,
            )

        draft = self.generator.generate(question, selected)
        if not draft.answerable:
            return self._abstain(
                missing_evidence=draft.missing_evidence,
                evidence=selected,
                searched_paper_ids=searched_paper_ids,
            )

        verification = self.verifier.verify(draft, selected)
        if not verification.valid:
            return self._abstain(
                missing_evidence=["The generated claims lacked valid source support."],
                evidence=selected,
                verification_errors=verification.errors,
                searched_paper_ids=searched_paper_ids,
            )

        return GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            answer=render_claims(draft),
            citations=verification.citations,
            evidence=selected,
            searched_paper_ids=searched_paper_ids,
        )

    def _abstain(
        self,
        *,
        missing_evidence: list[str],
        searched_paper_ids: list[str],
        evidence: list[Evidence] | None = None,
        verification_errors: list[str] | None = None,
    ) -> GroundedAnswer:
        """Build the single fail-closed refusal response."""
        return GroundedAnswer(
            status=AnswerStatus.ABSTAINED,
            answer=self.settings.refusal_message,
            evidence=evidence or [],
            missing_evidence=missing_evidence,
            verification_errors=verification_errors or [],
            searched_paper_ids=searched_paper_ids,
        )
