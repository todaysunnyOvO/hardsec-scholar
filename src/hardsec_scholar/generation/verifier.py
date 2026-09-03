"""Deterministically verify citation identifiers and source metadata."""

from hardsec_scholar.domain import Citation, Evidence
from hardsec_scholar.generation.models import AnswerDraft, VerificationResult


class CitationVerifier:
    """Reject unknown, duplicated, or otherwise invalid evidence bindings."""

    def verify(
        self, draft: AnswerDraft, selected_evidence: list[Evidence]
    ) -> VerificationResult:
        """Resolve every claim citation against the selected evidence only."""
        if not draft.answerable:
            return VerificationResult(
                valid=False,
                errors=["Cannot verify citations for an unanswerable draft"],
            )

        evidence_by_id = {item.id: item for item in selected_evidence}
        errors: list[str] = []
        citations: list[Citation] = []
        for claim_index, claim in enumerate(draft.claims, start=1):
            seen_for_claim: set[str] = set()
            for evidence_id in claim.evidence_ids:
                if evidence_id in seen_for_claim:
                    errors.append(
                        f"Claim {claim_index} repeats evidence ID: {evidence_id}"
                    )
                    continue
                seen_for_claim.add(evidence_id)
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    errors.append(
                        f"Claim {claim_index} cites unavailable evidence ID: {evidence_id}"
                    )
                    continue
                citations.append(
                    Citation(
                        evidence_id=item.id,
                        paper_id=item.paper_id,
                        paper_title=item.paper_title,
                        section=item.section,
                        page_start=item.page_start,
                        page_end=item.page_end,
                        claim=claim.text,
                    )
                )

        return VerificationResult(
            valid=not errors and bool(citations),
            citations=citations if not errors else [],
            errors=errors,
        )
