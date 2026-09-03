"""Protocols separating retrieval and model providers from RAG orchestration."""

from typing import Protocol

from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation.models import AnswerDraft


class EvidenceRetriever(Protocol):
    """Retrieve ranked local-paper evidence."""

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        """Return ranked evidence for a non-empty query."""
        ...


class AnswerGenerator(Protocol):
    """Generate claims constrained to supplied evidence identifiers."""

    def generate(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        """Return structured claims or a structured evidence-gap decision."""
        ...
