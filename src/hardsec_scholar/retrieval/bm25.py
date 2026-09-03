"""In-memory BM25 retrieval rebuilt from the SQLite source of truth."""

import re

from rank_bm25 import BM25Plus

from hardsec_scholar.domain import Evidence, PaperChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize English technical text without destroying compound identifiers."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    """Rank a small paper corpus with BM25+, which has stable positive IDF."""

    def __init__(self, chunks: list[PaperChunk]) -> None:
        """Build an in-memory index over the supplied ordered chunks."""
        self.chunks = chunks
        tokenized = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        self.index = BM25Plus(tokenized) if tokenized else None

    def search(
        self,
        query: str,
        *,
        top_k: int,
        paper_ids: list[str] | None = None,
    ) -> list[Evidence]:
        """Return positive-scoring chunks in descending BM25 order."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index is None:
            return []
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        allowed = set(paper_ids or [])
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if score > 0 and (not allowed or self.chunks[index].paper_id in allowed)
            ),
            key=lambda item: (-item[1], self.chunks[item[0]].id),
        )[:top_k]

        evidence: list[Evidence] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            evidence.append(
                Evidence(
                    id=f"evidence_{chunk.id}",
                    chunk_id=chunk.id,
                    paper_id=chunk.paper_id,
                    paper_title=chunk.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    bm25_rank=rank,
                    bm25_score=score,
                )
            )
        return evidence
