"""Tests for reciprocal-rank fusion."""

from hardsec_scholar.domain import Evidence
from hardsec_scholar.retrieval.fusion import reciprocal_rank_fusion


def _evidence(chunk_id: str, **ranks: int) -> Evidence:
    return Evidence(
        id=f"evidence_{chunk_id}",
        chunk_id=chunk_id,
        paper_id="paper",
        paper_title="Paper",
        page_start=1,
        page_end=1,
        text="Evidence",
        **ranks,
    )


def test_rrf_rewards_candidates_found_by_both_retrievers() -> None:
    dense = [_evidence("both", dense_rank=2), _evidence("dense", dense_rank=1)]
    lexical = [_evidence("both", bm25_rank=1)]

    fused = reciprocal_rank_fusion(dense, lexical)

    assert fused[0].chunk_id == "both"
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].fusion_score is not None
