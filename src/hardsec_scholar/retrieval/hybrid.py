"""Compose dense, lexical, fusion, and reranking stages."""

from hardsec_scholar.config import RetrievalSettings
from hardsec_scholar.domain import Evidence
from hardsec_scholar.retrieval.bm25 import BM25Retriever
from hardsec_scholar.retrieval.fusion import reciprocal_rank_fusion
from hardsec_scholar.retrieval.interfaces import Reranker
from hardsec_scholar.retrieval.vector import ChromaVectorIndex
from hardsec_scholar.storage import PaperRepository


class HybridRetriever:
    """Run the complete deterministic retrieval pipeline."""

    def __init__(
        self,
        *,
        repository: PaperRepository,
        vector_index: ChromaVectorIndex,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        """Store reusable adapters and validated retrieval settings."""
        self.repository = repository
        self.vector_index = vector_index
        self.reranker = reranker
        self.settings = settings

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        """Retrieve, fuse, and rerank evidence for one non-empty query."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        chunks = self.repository.list_chunks(paper_ids)
        lexical = BM25Retriever(chunks).search(
            query,
            top_k=self.settings.bm25_top_k,
            paper_ids=paper_ids,
        )
        dense = self.vector_index.search(
            query,
            top_k=self.settings.dense_top_k,
            paper_ids=paper_ids,
        )
        fused = reciprocal_rank_fusion(dense, lexical, rrf_k=self.settings.rrf_k)
        return self.reranker.rerank(
            query,
            fused,
            top_k=self.settings.rerank_top_k,
        )
