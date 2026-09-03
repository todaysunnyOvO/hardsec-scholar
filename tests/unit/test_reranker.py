"""Tests for reranking adapters without downloading model weights."""

from hardsec_scholar.domain import Evidence
from hardsec_scholar.retrieval.reranker import FlashRankReranker, IdentityReranker


def _candidate(chunk_id: str, score: float) -> Evidence:
    return Evidence(
        id=f"evidence_{chunk_id}",
        chunk_id=chunk_id,
        paper_id="paper",
        paper_title="Paper",
        page_start=1,
        page_end=1,
        text=chunk_id,
        fusion_score=score,
    )


class _FakeRanker:
    def rerank(self, request: object) -> list[dict[str, object]]:
        del request
        return [
            {"id": "second", "score": 0.9},
            {"id": "first", "score": 0.2},
        ]


def test_identity_reranker_preserves_order() -> None:
    candidates = [_candidate("first", 0.2), _candidate("second", 0.1)]

    results = IdentityReranker().rerank("query", candidates, top_k=1)

    assert [result.chunk_id for result in results] == ["first"]
    assert results[0].rerank_score == 0.2


def test_flashrank_adapter_uses_model_order() -> None:
    candidates = [_candidate("first", 0.2), _candidate("second", 0.1)]
    reranker = FlashRankReranker(ranker=_FakeRanker())

    results = reranker.rerank("query", candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["second", "first"]
    assert results[0].rerank_score == 0.9
