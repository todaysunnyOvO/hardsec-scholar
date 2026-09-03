"""Tests for page-aware PDF parsing and validation."""

from pathlib import Path

import pytest

from hardsec_scholar.ingestion.parser import PdfValidationError, parse_pdf


def test_parse_pdf_preserves_pages_and_hash(sample_paper_pdf: Path) -> None:
    parsed = parse_pdf(sample_paper_pdf)

    assert parsed.page_count == 2
    assert len(parsed.content_hash) == 64
    assert parsed.pages[0].page_number == 1
    assert "Hardware Fuzzing" in parsed.pages[0].text


def test_parser_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "paper.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(PdfValidationError, match="Only PDF"):
        parse_pdf(text_file)
