"""Tests for deterministic paper metadata extraction."""

from pathlib import Path

from hardsec_scholar.domain import ResearchArea
from hardsec_scholar.ingestion.metadata import extract_metadata
from hardsec_scholar.ingestion.parser import parse_pdf


def test_extract_hardware_fuzzing_metadata(sample_paper_pdf: Path) -> None:
    metadata = extract_metadata(parse_pdf(sample_paper_pdf))

    assert metadata.title == "Coverage-Guided Hardware Fuzzing for Secure Processors"
    assert metadata.authors == ("Alice Smith", "Bob Jones")
    assert metadata.year == 2025
    assert metadata.doi == "10.1234/example.2025"
    assert metadata.research_area is ResearchArea.HARDWARE_FUZZING
