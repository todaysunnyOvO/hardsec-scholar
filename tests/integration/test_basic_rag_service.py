"""Integration tests for one-pass grounded RAG and fail-closed refusal."""

from hardsec_scholar.config import GenerationSettings
from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation import (
    AnswerDraft,
    AnswerStatus,
    BasicRAGService,
    ClaimDraft,
)


class _FakeRetriever:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self.paper_ids: list[str] | None = None

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        self.paper_ids = paper_ids
        return self.evidence


class _FakeGenerator:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls = 0

    def generate(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        self.calls += 1
        return self.draft


def _evidence() -> Evidence:
    return Evidence(
        id="E1",
        chunk_id="C1",
        paper_id="P1",
        paper_title="Hardware Fuzzing",
        section="Evaluation",
        page_start=8,
        page_end=9,
        text="Coverage feedback guides mutation selection.",
    )


def _service(
    evidence: list[Evidence], draft: AnswerDraft
) -> tuple[BasicRAGService, _FakeRetriever, _FakeGenerator]:
    retriever = _FakeRetriever(evidence)
    generator = _FakeGenerator(draft)
    service = BasicRAGService(
        retriever=retriever,
        generator=generator,
        settings=GenerationSettings(),
    )
    return service, retriever, generator


def test_basic_rag_returns_inline_id_and_resolved_citation() -> None:
    service, retriever, _ = _service(
        [_evidence()],
        AnswerDraft(
            answerable=True,
            claims=[
                ClaimDraft(
                    text="Coverage feedback guides mutation selection.",
                    evidence_ids=["E1"],
                )
            ],
        ),
    )

    result = service.answer("What guides mutation selection?", paper_ids=["P1"])

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer.endswith("[E1]")
    assert result.citations[0].page_start == 8
    assert result.evidence[0].text.startswith("Coverage feedback")
    assert result.searched_paper_ids == ["P1"]
    assert retriever.paper_ids == ["P1"]


def test_basic_rag_refuses_without_retrieved_evidence_or_model_call() -> None:
    service, _, generator = _service(
        [],
        AnswerDraft(
            answerable=False,
            missing_evidence=["An evaluation result is missing."],
        ),
    )

    result = service.answer("What is the overhead?")

    assert result.status is AnswerStatus.ABSTAINED
    assert result.answer.startswith("The indexed papers do not provide enough evidence")
    assert result.missing_evidence == ["No relevant evidence was retrieved."]
    assert generator.calls == 0


def test_basic_rag_refuses_when_model_reports_evidence_gap() -> None:
    service, _, _ = _service(
        [_evidence()],
        AnswerDraft(
            answerable=False,
            missing_evidence=["No latency measurement is present."],
        ),
    )

    result = service.answer("What is the measured latency?")

    assert result.status is AnswerStatus.ABSTAINED
    assert result.missing_evidence == ["No latency measurement is present."]
    assert result.evidence[0].id == "E1"


def test_basic_rag_refuses_fabricated_citation_id() -> None:
    service, _, _ = _service(
        [_evidence()],
        AnswerDraft(
            answerable=True,
            claims=[ClaimDraft(text="A fabricated claim.", evidence_ids=["E999"])],
        ),
    )

    result = service.answer("Give me a fact.")

    assert result.status is AnswerStatus.ABSTAINED
    assert result.citations == []
    assert "E999" in result.verification_errors[0]
