"""Dense-only retrieval adapter used by the evaluation baseline."""

from hardsec_scholar.domain import Evidence
from hardsec_scholar.retrieval.vector import ChromaVectorIndex


class DenseRetriever:
    """Expose one vector search as the minimal RAG retrieval baseline."""

    def __init__(self, vector_index: ChromaVectorIndex, *, top_k: int = 10) -> None:
        """Store the vector index and a positive retrieval cutoff."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.vector_index = vector_index
        self.top_k = top_k

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        """Return dense candidates for one non-empty question."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        return self.vector_index.search(
            query,
            top_k=self.top_k,
            paper_ids=paper_ids,
        )
