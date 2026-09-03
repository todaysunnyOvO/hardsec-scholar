"""Tests for technical-term-preserving BM25 retrieval."""

from hardsec_scholar.domain import PaperChunk
from hardsec_scholar.retrieval.bm25 import BM25Retriever, tokenize_for_bm25


def _chunks() -> list[PaperChunk]:
    return [
        PaperChunk(
            id="p1_chunk_0000",
            paper_id="p1",
            title="Fuzzer",
            section="Evaluation",
            page_start=1,
            page_end=1,
            chunk_index=0,
            text="RTL branch coverage guides hardware fuzzing input mutation.",
        ),
        PaperChunk(
            id="p2_chunk_0000",
            paper_id="p2",
            title="Side Channel",
            section="Method",
            page_start=2,
            page_end=2,
            chunk_index=0,
            text="Correlation power analysis recovers an AES key from traces.",
        ),
    ]


def test_tokenizer_preserves_compound_terms() -> None:
    assert tokenize_for_bm25("coverage-guided RTL_fuzzing") == [
        "coverage-guided",
        "rtl_fuzzing",
    ]


def test_bm25_ranks_exact_hardware_term() -> None:
    results = BM25Retriever(_chunks()).search("RTL coverage", top_k=2)

    assert results[0].paper_id == "p1"
    assert results[0].bm25_rank == 1
    assert results[0].bm25_score is not None


def test_bm25_respects_paper_filter() -> None:
    results = BM25Retriever(_chunks()).search(
        "coverage traces", top_k=2, paper_ids=["p2"]
    )

    assert all(result.paper_id == "p2" for result in results)
