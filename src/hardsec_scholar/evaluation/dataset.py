"""Load and validate evaluation JSONL against the indexed paper corpus."""

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hardsec_scholar.evaluation.models import EvaluationCategory, EvaluationSample
from hardsec_scholar.storage import PaperRepository

EXPECTED_CATEGORY_COUNTS = {
    EvaluationCategory.SINGLE_PAPER_FACT: 10,
    EvaluationCategory.MECHANISM: 5,
    EvaluationCategory.THREAT_MODEL: 4,
    EvaluationCategory.EXPERIMENT: 4,
    EvaluationCategory.COMPARISON: 4,
    EvaluationCategory.UNANSWERABLE: 3,
}


class DatasetValidationReport(BaseModel):
    """Summarize a successfully validated evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=0)
    answerable_count: int = Field(ge=0)
    abstention_count: int = Field(ge=0)
    category_counts: dict[str, int]
    covered_paper_ids: list[str]


def load_evaluation_dataset(path: Path | str) -> list[EvaluationSample]:
    """Load UTF-8 JSONL records and report malformed line numbers."""
    dataset_path = Path(path)
    samples: list[EvaluationSample] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                samples.append(EvaluationSample.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation sample at line {line_number}: {exc}"
                ) from exc
    return samples


def validate_evaluation_dataset(
    samples: list[EvaluationSample],
    repository: PaperRepository,
    *,
    require_benchmark_shape: bool = True,
) -> DatasetValidationReport:
    """Verify uniqueness, category quotas, and every gold corpus reference."""
    ids = [sample.id for sample in samples]
    questions = [sample.question.casefold().strip() for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation sample IDs must be unique")
    if len(questions) != len(set(questions)):
        raise ValueError("Evaluation questions must be unique")

    counts = Counter(sample.category for sample in samples)
    if require_benchmark_shape:
        if not 30 <= len(samples) <= 50:
            raise ValueError("The benchmark must contain between 30 and 50 samples")
        if counts != Counter(EXPECTED_CATEGORY_COUNTS):
            raise ValueError(
                "Category counts do not match the approved 30-question benchmark shape"
            )

    covered_papers: set[str] = set()
    for sample in samples:
        for paper_id in sample.paper_ids:
            if repository.get_paper(paper_id) is None:
                raise ValueError(f"{sample.id} references unknown paper {paper_id}")
            covered_papers.add(paper_id)

        evidence_pages: set[int] = set()
        for chunk_id in sample.relevant_chunk_ids:
            chunk = repository.get_chunk(chunk_id)
            if chunk is None:
                raise ValueError(f"{sample.id} references unknown chunk {chunk_id}")
            if chunk.paper_id not in sample.paper_ids:
                raise ValueError(
                    f"{sample.id} chunk {chunk_id} is outside its gold papers"
                )
            evidence_pages.update(range(chunk.page_start, chunk.page_end + 1))
        if evidence_pages != set(sample.expected_pages):
            raise ValueError(
                f"{sample.id} expected_pages do not match its gold chunk ranges"
            )

    if require_benchmark_shape:
        corpus_ids = {paper.id for paper in repository.list_papers()}
        if covered_papers != corpus_ids:
            missing = sorted(corpus_ids - covered_papers)
            raise ValueError(f"Benchmark does not cover all indexed papers: {missing}")

    return DatasetValidationReport(
        sample_count=len(samples),
        answerable_count=sum(not sample.should_abstain for sample in samples),
        abstention_count=sum(sample.should_abstain for sample in samples),
        category_counts={category.value: counts[category] for category in counts},
        covered_paper_ids=sorted(covered_papers),
    )
