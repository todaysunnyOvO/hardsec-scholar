"""Deterministic metrics for retrieval, citations, refusal, and answer overlap."""

import re
from collections import Counter

from hardsec_scholar.domain import Evidence
from hardsec_scholar.evaluation.models import (
    AnswerMetrics,
    EvaluationSample,
    RetrievalMetrics,
)
from hardsec_scholar.generation import AnswerStatus, GroundedAnswer

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def unique_evidence(items: list[Evidence]) -> list[Evidence]:
    """Deduplicate ranked evidence without changing first-seen order."""
    result: list[Evidence] = []
    seen: set[str] = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        result.append(item)
    return result


def evaluate_retrieval(
    sample: EvaluationSample,
    first_search: list[Evidence],
    all_searches: list[Evidence],
) -> RetrievalMetrics:
    """Measure initial ranking and all-round evidence coverage."""
    first = unique_evidence(first_search)
    all_items = unique_evidence(all_searches)
    if sample.should_abstain:
        return RetrievalMetrics(retrieved_chunk_count=len(all_items))

    gold_chunks = set(sample.relevant_chunk_ids)
    first_ids = [item.chunk_id for item in first]
    all_ids = {item.chunk_id for item in all_items}
    first_rank = next(
        (rank for rank, chunk_id in enumerate(first_ids, start=1) if chunk_id in gold_chunks),
        None,
    )
    expected_pages = set(sample.expected_pages)
    return RetrievalMetrics(
        recall_at_5=len(gold_chunks.intersection(first_ids[:5])) / len(gold_chunks),
        recall_at_10=len(gold_chunks.intersection(first_ids[:10])) / len(gold_chunks),
        mrr=1.0 / first_rank if first_rank is not None else 0.0,
        first_relevant_rank=first_rank,
        paper_hit=any(item.paper_id in sample.paper_ids for item in first[:10]),
        page_hit=any(
            item.paper_id in sample.paper_ids
            and expected_pages.intersection(range(item.page_start, item.page_end + 1))
            for item in first[:10]
        ),
        all_round_recall=len(gold_chunks.intersection(all_ids)) / len(gold_chunks),
        retrieved_chunk_count=len(all_items),
    )


def token_f1(reference: str, candidate: str) -> float:
    """Compute bag-of-token F1 as a transparent lexical similarity signal."""
    reference_tokens = TOKEN_PATTERN.findall(reference.casefold())
    candidate_tokens = TOKEN_PATTERN.findall(candidate.casefold())
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(
    sample: EvaluationSample,
    answer: GroundedAnswer,
) -> AnswerMetrics:
    """Measure refusal correctness and exact gold citation agreement."""
    abstained = answer.status is AnswerStatus.ABSTAINED
    abstention_correct = abstained == sample.should_abstain
    if sample.should_abstain:
        return AnswerMetrics(
            abstention_correct=abstention_correct,
            grounded=abstained and not answer.citations,
            citation_count=len(answer.citations),
        )

    evidence_chunks = {item.id: item.chunk_id for item in answer.evidence}
    cited_chunk_ids = {
        evidence_chunks[citation.evidence_id]
        for citation in answer.citations
        if citation.evidence_id in evidence_chunks
    }
    gold_chunks = set(sample.relevant_chunk_ids)
    citation_count = len(answer.citations)
    correct_citations = sum(
        evidence_chunks.get(citation.evidence_id) in gold_chunks
        for citation in answer.citations
    )
    expected_pages = set(sample.expected_pages)
    page_correct = sum(
        bool(expected_pages.intersection(range(citation.page_start, citation.page_end + 1)))
        for citation in answer.citations
    )
    paper_correct = sum(
        citation.paper_id in sample.paper_ids for citation in answer.citations
    )
    return AnswerMetrics(
        abstention_correct=abstention_correct,
        reference_token_f1=(
            token_f1(sample.reference_answer, answer.answer) if not abstained else 0.0
        ),
        citation_precision=(
            correct_citations / citation_count if citation_count else 0.0
        ),
        citation_recall=len(gold_chunks.intersection(cited_chunk_ids)) / len(gold_chunks),
        citation_page_precision=page_correct / citation_count if citation_count else 0.0,
        citation_paper_precision=(
            paper_correct / citation_count if citation_count else 0.0
        ),
        grounded=(
            not abstained
            and citation_count > 0
            and not answer.verification_errors
        ),
        citation_count=citation_count,
    )
