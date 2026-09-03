"""Shared test fixtures for deterministic PDF ingestion tests."""

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def sample_paper_pdf(tmp_path: Path) -> Path:
    """Create a small text PDF resembling an English hardware-security paper."""
    output = tmp_path / "sample-paper.pdf"
    document = pymupdf.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "Coverage-Guided Hardware Fuzzing for Secure Processors",
        fontsize=18,
    )
    first_page.insert_text((72, 105), "Alice Smith and Bob Jones", fontsize=11)
    first_page.insert_text(
        (72, 145),
        "Abstract",
        fontsize=13,
    )
    first_page.insert_textbox(
        (72, 160, 520, 260),
        "We present a hardware fuzzing method that uses RTL coverage feedback "
        "to discover security violations in processor designs. DOI: 10.1234/example.2025",
        fontsize=10,
    )
    first_page.insert_text((72, 285), "I. Introduction", fontsize=13)
    first_page.insert_textbox(
        (72, 300, 520, 430),
        "Coverage-guided fuzzing has been successful in software. Our method adapts "
        "fuzz testing to register-transfer level processor implementations in 2025.",
        fontsize=10,
    )

    second_page = document.new_page()
    second_page.insert_text((72, 72), "IV. Evaluation", fontsize=13)
    second_page.insert_textbox(
        (72, 90, 520, 230),
        "The evaluation uses an open processor core. RTL branch coverage guides input "
        "mutation, and the experiment reports newly discovered security violations.",
        fontsize=10,
    )
    second_page.insert_text((72, 260), "References", fontsize=13)
    second_page.insert_textbox(
        (72, 280, 520, 380),
        "[1] A reference that should be excluded from the searchable corpus.",
        fontsize=10,
    )

    document.save(output)
    document.close()
    return output
