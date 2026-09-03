"""Tests for evidence-grounded evaluation dataset validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hardsec_scholar.domain import PaperChunk, PaperMetadata
from hardsec_scholar.evaluation import (
    EvaluationCategory,
    EvaluationSample,
    validate_evaluation_dataset,
)
from hardsec_scholar.storage import PaperRepository


def _repository(tmp_path: Path) -> PaperRepository:
    repository = PaperRepository(tmp_path / "papers.db")
    repository.initialize()
    paper = PaperMetadata(
        id="paper_1",
        content_hash="abc123",
        title="Paper One",
        file_path=tmp_path / "paper.pdf",
        page_count=5,
    )
    chunk = PaperChunk(
        id="paper_1_chunk_0001",
        paper_id=paper.id,
        title=paper.title,
        section="Evaluation",
        page_start=2,
        page_end=3,
        chunk_index=1,
        text="The experiment detected a speculative leak.",
    )
    repository.save_paper(paper, [chunk])
    return repository


def test_validate_sample_against_repository(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    sample = EvaluationSample(
        id="hardsec_eval_001",
        question="What did the experiment detect?",
        category=EvaluationCategory.SINGLE_PAPER_FACT,
        paper_ids=["paper_1"],
        reference_answer="It detected a speculative leak.",
        relevant_chunk_ids=["paper_1_chunk_0001"],
        expected_pages=[2, 3],
    )

    report = validate_evaluation_dataset(
        [sample], repository, require_benchmark_shape=False
    )

    assert report.sample_count == 1
    assert report.answerable_count == 1
    assert report.covered_paper_ids == ["paper_1"]


def test_reject_page_ranges_that_do_not_match_chunks(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    sample = EvaluationSample(
        id="hardsec_eval_001",
        question="What did the experiment detect?",
        category=EvaluationCategory.SINGLE_PAPER_FACT,
        paper_ids=["paper_1"],
        reference_answer="It detected a speculative leak.",
        relevant_chunk_ids=["paper_1_chunk_0001"],
        expected_pages=[2],
    )

    with pytest.raises(ValueError, match="expected_pages"):
        validate_evaluation_dataset(
            [sample], repository, require_benchmark_shape=False
        )


def test_unanswerable_sample_cannot_declare_gold_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot declare gold"):
        EvaluationSample(
            id="hardsec_eval_001",
            question="What fabrication node was used?",
            category=EvaluationCategory.UNANSWERABLE,
            paper_ids=["paper_1"],
            reference_answer="The corpus does not say.",
            relevant_chunk_ids=[],
            expected_pages=[],
            should_abstain=True,
        )
