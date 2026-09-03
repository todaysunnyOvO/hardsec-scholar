"""Tests for deterministic citation integrity verification."""

from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation import AnswerDraft, CitationVerifier, ClaimDraft


def _evidence() -> Evidence:
    return Evidence(
        id="E1",
        chunk_id="C1",
        paper_id="P1",
        paper_title="Hardware Fuzzing",
        section="Evaluation",
        page_start=8,
        page_end=9,
        text="Coverage feedback improves mutation selection.",
    )


def test_verifier_resolves_valid_citation_metadata() -> None:
    draft = AnswerDraft(
        answerable=True,
        claims=[ClaimDraft(text="Coverage guides mutations.", evidence_ids=["E1"])],
    )

    result = CitationVerifier().verify(draft, [_evidence()])

    assert result.valid is True
    assert result.errors == []
    assert result.citations[0].paper_title == "Hardware Fuzzing"
    assert result.citations[0].page_start == 8
    assert result.citations[0].claim == "Coverage guides mutations."


def test_verifier_rejects_unavailable_evidence_id() -> None:
    draft = AnswerDraft(
        answerable=True,
        claims=[ClaimDraft(text="Unsupported claim.", evidence_ids=["E999"])],
    )

    result = CitationVerifier().verify(draft, [_evidence()])

    assert result.valid is False
    assert result.citations == []
    assert "E999" in result.errors[0]


def test_verifier_rejects_duplicate_binding_within_claim() -> None:
    draft = AnswerDraft(
        answerable=True,
        claims=[ClaimDraft(text="A claim.", evidence_ids=["E1", "E1"])],
    )

    result = CitationVerifier().verify(draft, [_evidence()])

    assert result.valid is False
    assert "repeats" in result.errors[0]
