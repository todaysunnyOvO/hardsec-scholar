"""Fuse independently ranked retrieval results."""

from hardsec_scholar.domain import Evidence


def reciprocal_rank_fusion(
    dense: list[Evidence],
    lexical: list[Evidence],
    *,
    rrf_k: int = 60,
) -> list[Evidence]:
    """Merge dense and lexical rankings with deterministic RRF ordering."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    merged: dict[str, Evidence] = {}
    scores: dict[str, float] = {}
    for candidates, rank_field in (
        (dense, "dense_rank"),
        (lexical, "bm25_rank"),
    ):
        for candidate in candidates:
            rank = getattr(candidate, rank_field)
            if rank is None:
                continue
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            previous = merged.get(candidate.chunk_id)
            if previous is None:
                merged[candidate.chunk_id] = candidate
            else:
                merged[candidate.chunk_id] = previous.model_copy(
                    update={
                        "dense_rank": previous.dense_rank or candidate.dense_rank,
                        "dense_score": previous.dense_score or candidate.dense_score,
                        "bm25_rank": previous.bm25_rank or candidate.bm25_rank,
                        "bm25_score": previous.bm25_score or candidate.bm25_score,
                    }
                )

    fused = [
        candidate.model_copy(update={"fusion_score": scores[chunk_id]})
        for chunk_id, candidate in merged.items()
    ]
    return sorted(
        fused,
        key=lambda candidate: (-(candidate.fusion_score or 0.0), candidate.chunk_id),
    )
