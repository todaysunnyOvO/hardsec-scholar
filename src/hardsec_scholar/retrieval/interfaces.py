"""Protocols shared by retrieval providers."""

from typing import Protocol

from hardsec_scholar.domain import Evidence


class EmbeddingProvider(Protocol):
    """Abstract document and query embedding generation."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus documents in input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one user query."""
        ...


class Reranker(Protocol):
    """Abstract candidate reranking."""

    def rerank(
        self, query: str, candidates: list[Evidence], *, top_k: int
    ) -> list[Evidence]:
        """Return candidates in descending relevance order."""
        ...
