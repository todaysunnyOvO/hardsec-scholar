"""Tests for deterministic benchmark metrics."""

import pytest

from hardsec_scholar.domain import Citation, Evidence
from hardsec_scholar.evaluation import (
    EvaluationCategory,
    EvaluationSample,
    evaluate_answer,
    evaluate_retrieval,
    token_f1,
)
from hardsec_scholar.generation import AnswerStatus, GroundedAnswer


def _evidence(chunk_id: str, paper_id: str, page: int) -> Evidence:
    return Evidence(
        id=f"evidence_{chunk_id}",
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title="Paper",
        page_start=page,
        page_end=page,
        text="Supported evidence.",
    )


def _sample() -> EvaluationSample:
    return EvaluationSample(
        id="hardsec_eval_001",
        question="What mechanism is used?",
        category=EvaluationCategory.MECHANISM,
        paper_ids=["paper_1"],
        reference_answer="The mechanism uses coverage feedback.",
        relevant_chunk_ids=["chunk_gold_1", "chunk_gold_2"],
        expected_pages=[3, 4],
    )


def test_retrieval_metrics_use_first_search_and_all_round_union() -> None:
    sample = _sample()
    first = [
        _evidence("chunk_other", "paper_2", 1),
        _evidence("chunk_gold_1", "paper_1", 3),
    ]
    later = [*first, _evidence("chunk_gold_2", "paper_1", 4)]

    metrics = evaluate_retrieval(sample, first, later)

    assert metrics.recall_at_5 == 0.5
    assert metrics.mrr == 0.5
    assert metrics.first_relevant_rank == 2
    assert metrics.paper_hit is True
    assert metrics.page_hit is True
    assert metrics.all_round_recall == 1.0


def test_answer_metrics_resolve_citations_to_gold_chunks() -> None:
    sample = _sample()
    evidence = [_evidence("chunk_gold_1", "paper_1", 3)]
    answer = GroundedAnswer(
        status=AnswerStatus.ANSWERED,
        answer="The mechanism uses coverage feedback.",
        evidence=evidence,
        citations=[
            Citation(
                evidence_id=evidence[0].id,
                paper_id="paper_1",
                paper_title="Paper",
                page_start=3,
                page_end=3,
                claim="The mechanism uses coverage feedback.",
            )
        ],
    )

    metrics = evaluate_answer(sample, answer)

    assert metrics.abstention_correct is True
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 0.5
    assert metrics.citation_page_precision == 1.0
    assert metrics.citation_paper_precision == 1.0
    assert metrics.grounded is True
    assert metrics.reference_token_f1 == pytest.approx(1.0)


def test_unanswerable_metrics_reward_clean_refusal() -> None:
    sample = EvaluationSample(
        id="hardsec_eval_030",
        question="What unrelated result is reported?",
        category=EvaluationCategory.UNANSWERABLE,
        reference_answer="The corpus does not provide this information.",
        should_abstain=True,
    )
    answer = GroundedAnswer(
        status=AnswerStatus.ABSTAINED,
        answer="The corpus does not provide this information.",
    )

    metrics = evaluate_answer(sample, answer)

    assert metrics.abstention_correct is True
    assert metrics.grounded is True
    assert metrics.citation_precision is None


def test_token_f1_is_zero_without_shared_tokens() -> None:
    assert token_f1("alpha beta", "gamma delta") == 0.0
