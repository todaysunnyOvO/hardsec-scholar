"""Candidate reranking adapters."""

from pathlib import Path
from typing import Any

from flashrank import Ranker, RerankRequest

from hardsec_scholar.domain import Evidence


class IdentityReranker:
    """Keep RRF ordering when model reranking is disabled."""

    def rerank(
        self, query: str, candidates: list[Evidence], *, top_k: int
    ) -> list[Evidence]:
        """Return the first candidates and expose fusion score as final score."""
        del query
        return [
            candidate.model_copy(update={"rerank_score": candidate.fusion_score})
            for candidate in candidates[:top_k]
        ]


class FlashRankReranker:
    """Apply a lightweight local cross-encoder through FlashRank."""

    def __init__(
        self,
        *,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: Path | str = "data/models/flashrank",
        max_length: int = 512,
        ranker: Any | None = None,
    ) -> None:
        """Initialize lazily injectable FlashRank state."""
        self.ranker = ranker or Ranker(
            model_name=model_name,
            cache_dir=str(cache_dir),
            max_length=max_length,
        )

    def rerank(
        self, query: str, candidates: list[Evidence], *, top_k: int
    ) -> list[Evidence]:
        """Rerank candidates and preserve their retrieval diagnostics."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        passages = [
            {"id": candidate.chunk_id, "text": candidate.text, "meta": {}}
            for candidate in candidates
        ]
        results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        reranked: list[Evidence] = []
        for result in results[:top_k]:
            chunk_id = str(result["id"])
            candidate = by_id.get(chunk_id)
            if candidate is None:
                continue
            reranked.append(
                candidate.model_copy(update={"rerank_score": float(result["score"])})
            )
        return reranked
