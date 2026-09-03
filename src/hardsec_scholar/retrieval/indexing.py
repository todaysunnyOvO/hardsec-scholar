"""Promote parsed papers into the dense retrieval index."""

from hardsec_scholar.domain import PaperStatus
from hardsec_scholar.retrieval.vector import ChromaVectorIndex
from hardsec_scholar.storage import PaperRepository


class PaperIndexingService:
    """Coordinate vector indexing and paper lifecycle status."""

    def __init__(
        self, repository: PaperRepository, vector_index: ChromaVectorIndex
    ) -> None:
        """Store local persistence and vector adapters."""
        self.repository = repository
        self.vector_index = vector_index

    def index_paper(self, paper_id: str) -> int:
        """Embed all parsed chunks and mark the paper as indexed."""
        paper = self.repository.get_paper(paper_id)
        if paper is None:
            raise KeyError(f"Unknown paper: {paper_id}")
        chunks = self.repository.get_chunks(paper_id)
        if not chunks:
            raise ValueError(f"Paper has no chunks: {paper_id}")
        indexed_count = self.vector_index.upsert(chunks)
        self.repository.update_status(paper_id, PaperStatus.INDEXED)
        return indexed_count

    def remove_paper(self, paper_id: str) -> bool:
        """Delete vectors before removing the source paper and chunks."""
        self.vector_index.delete_paper(paper_id)
        return self.repository.delete_paper(paper_id)
