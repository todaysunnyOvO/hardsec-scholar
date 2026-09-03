"""Integration tests for vector indexing and hybrid retrieval."""

import hashlib
from pathlib import Path

from hardsec_scholar.config import GenerationSettings, RetrievalSettings
from hardsec_scholar.domain import Evidence, PaperChunk, PaperMetadata, PaperStatus
from hardsec_scholar.generation import (
    AnswerDraft,
    AnswerStatus,
    BasicRAGService,
    ClaimDraft,
)
from hardsec_scholar.retrieval.hybrid import HybridRetriever
from hardsec_scholar.retrieval.indexing import PaperIndexingService
from hardsec_scholar.retrieval.reranker import IdentityReranker
from hardsec_scholar.retrieval.vector import ChromaVectorIndex
from hardsec_scholar.storage import PaperRepository


class _KeywordEmbeddingProvider:
    terms = ("fuzzing", "coverage", "power", "cache")

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in self.terms]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class _EvidenceEchoGenerator:
    def generate(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        return AnswerDraft(
            answerable=True,
            claims=[
                ClaimDraft(
                    text="Coverage feedback guides hardware fuzzing mutations.",
                    evidence_ids=[evidence[0].id],
                )
            ],
        )


def _save_paper(
    repository: PaperRepository, paper_id: str, title: str, text: str
) -> None:
    paper = PaperMetadata(
        id=paper_id,
        content_hash=hashlib.sha256(paper_id.encode()).hexdigest(),
        title=title,
        file_path=Path(f"{paper_id}.pdf"),
        page_count=1,
        status=PaperStatus.PARSED,
    )
    chunk = PaperChunk(
        id=f"{paper_id}_chunk_0000",
        paper_id=paper_id,
        title=title,
        section="Evaluation",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text=text,
    )
    repository.save_paper(paper, [chunk])


def test_index_and_hybrid_search(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "papers.db")
    repository.initialize()
    _save_paper(
        repository,
        "fuzz-paper",
        "Hardware Fuzzing",
        "RTL coverage feedback guides hardware fuzzing mutations.",
    )
    _save_paper(
        repository,
        "sca-paper",
        "Power Analysis",
        "Power traces reveal cryptographic key leakage.",
    )
    vector_index = ChromaVectorIndex(
        tmp_path / "chroma", _KeywordEmbeddingProvider(), collection_name="test"
    )
    indexing = PaperIndexingService(repository, vector_index)
    assert indexing.index_paper("fuzz-paper") == 1
    assert indexing.index_paper("sca-paper") == 1

    retriever = HybridRetriever(
        repository=repository,
        vector_index=vector_index,
        reranker=IdentityReranker(),
        settings=RetrievalSettings(
            dense_top_k=2,
            bm25_top_k=2,
            rerank_top_k=2,
        ),
    )
    results = retriever.search("RTL coverage fuzzing")

    assert results[0].paper_id == "fuzz-paper"
    assert results[0].dense_rank is not None
    assert results[0].bm25_rank is not None
    assert repository.get_paper("fuzz-paper").status is PaperStatus.INDEXED


def test_hybrid_search_respects_selected_papers(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "papers.db")
    repository.initialize()
    _save_paper(repository, "p1", "One", "RTL coverage fuzzing")
    _save_paper(repository, "p2", "Two", "Power analysis traces")
    vector_index = ChromaVectorIndex(
        tmp_path / "chroma", _KeywordEmbeddingProvider(), collection_name="filtered"
    )
    indexing = PaperIndexingService(repository, vector_index)
    indexing.index_paper("p1")
    indexing.index_paper("p2")
    retriever = HybridRetriever(
        repository=repository,
        vector_index=vector_index,
        reranker=IdentityReranker(),
        settings=RetrievalSettings(dense_top_k=2, bm25_top_k=2, rerank_top_k=2),
    )

    results = retriever.search("coverage power", paper_ids=["p2"])

    assert results
    assert {result.paper_id for result in results} == {"p2"}


def test_hybrid_retrieval_flows_into_grounded_answer(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "papers.db")
    repository.initialize()
    _save_paper(
        repository,
        "fuzz-paper",
        "Hardware Fuzzing",
        "RTL coverage feedback guides hardware fuzzing mutations.",
    )
    vector_index = ChromaVectorIndex(
        tmp_path / "chroma", _KeywordEmbeddingProvider(), collection_name="rag"
    )
    PaperIndexingService(repository, vector_index).index_paper("fuzz-paper")
    retriever = HybridRetriever(
        repository=repository,
        vector_index=vector_index,
        reranker=IdentityReranker(),
        settings=RetrievalSettings(dense_top_k=2, bm25_top_k=2, rerank_top_k=2),
    )
    service = BasicRAGService(
        retriever=retriever,
        generator=_EvidenceEchoGenerator(),
        settings=GenerationSettings(),
    )

    result = service.answer("What guides hardware fuzzing mutations?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.citations[0].paper_id == "fuzz-paper"
    assert result.citations[0].section == "Evaluation"
    assert result.citations[0].page_start == 1
    assert f"[{result.evidence[0].id}]" in result.answer
