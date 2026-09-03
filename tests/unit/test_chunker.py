"""Tests for section-aware and page-preserving chunking."""

from pathlib import Path

from hardsec_scholar.ingestion.chunker import chunk_document
from hardsec_scholar.ingestion.parser import parse_pdf


def test_chunker_detects_sections_and_excludes_references(
    sample_paper_pdf: Path,
) -> None:
    parsed = parse_pdf(sample_paper_pdf)

    chunks = chunk_document(
        parsed,
        paper_id="paper-1",
        title="Test Paper",
        chunk_size_tokens=100,
        overlap_tokens=10,
    )

    sections = {chunk.section for chunk in chunks}
    assert "Abstract" in sections
    assert "Introduction" in sections
    assert "Evaluation" in sections
    assert "References" not in sections
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
