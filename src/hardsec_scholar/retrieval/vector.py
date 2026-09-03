"""Persistent dense-vector indexing with Chroma."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from hardsec_scholar.domain import Evidence, PaperChunk
from hardsec_scholar.retrieval.interfaces import EmbeddingProvider


class ChromaVectorIndex:
    """Own one long-lived Chroma client and cosine collection."""

    def __init__(
        self,
        path: Path | str,
        embedding_provider: EmbeddingProvider,
        *,
        collection_name: str = "hardsec_papers",
        client: Any | None = None,
    ) -> None:
        """Initialize one persistent collection without a built-in embedder."""
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.client = client or chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def upsert(self, chunks: list[PaperChunk], *, batch_size: int = 64) -> int:
        """Embed and upsert chunks in bounded batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedding_provider.embed_documents(
                [chunk.text for chunk in batch]
            )
            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding provider returned an unexpected result count"
                )
            chroma_embeddings: list[Sequence[float] | Sequence[int]] = [
                embedding for embedding in embeddings
            ]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=chroma_embeddings,
                documents=[chunk.text for chunk in batch],
                metadatas=[self._metadata(chunk) for chunk in batch],
            )
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        paper_ids: list[str] | None = None,
    ) -> list[Evidence]:
        """Search by a supplied query embedding and optional paper filter."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        count = self.collection.count()
        if count == 0:
            return []
        where = self._paper_filter(paper_ids)
        query_embedding: Sequence[float] = self.embedding_provider.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        evidence: list[Evidence] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            if document is None or metadata is None:
                continue
            evidence.append(
                Evidence(
                    id=f"evidence_{chunk_id}",
                    chunk_id=chunk_id,
                    paper_id=str(metadata["paper_id"]),
                    paper_title=str(metadata["title"]),
                    section=str(metadata["section"]) or None,
                    page_start=int(metadata["page_start"]),
                    page_end=int(metadata["page_end"]),
                    text=document,
                    dense_rank=rank,
                    dense_score=1.0 - float(distance),
                )
            )
        return evidence

    def delete_paper(self, paper_id: str) -> None:
        """Remove all vectors associated with one paper."""
        self.collection.delete(where={"paper_id": paper_id})

    @staticmethod
    def _metadata(chunk: PaperChunk) -> dict[str, str | int]:
        return {
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "section": chunk.section or "",
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "source_type": chunk.source_type,
        }

    @staticmethod
    def _paper_filter(paper_ids: list[str] | None) -> dict[str, Any] | None:
        if not paper_ids:
            return None
        if len(paper_ids) == 1:
            return {"paper_id": paper_ids[0]}
        return {"paper_id": {"$in": paper_ids}}
