"""Tests for shared domain entities."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hardsec_scholar.domain import (
    HardSecEvidence,
    PaperChunk,
    PaperMetadata,
    ResearchArea,
)


def test_paper_metadata_defaults_to_pending() -> None:
    paper = PaperMetadata(
        id="paper-1",
        content_hash="sha256:abc",
        title="A Hardware Security Paper",
        research_area=ResearchArea.ARCHITECTURAL_SECURITY,
        file_path=Path("data/papers/paper-1.pdf"),
        page_count=12,
    )

    assert paper.status == "pending"
    assert paper.research_area is ResearchArea.ARCHITECTURAL_SECURITY


def test_chunk_rejects_reversed_page_range() -> None:
    with pytest.raises(ValidationError, match="page_end"):
        PaperChunk(
            id="chunk-1",
            paper_id="paper-1",
            title="Paper",
            page_start=5,
            page_end=4,
            chunk_index=0,
            text="Evidence text",
        )


def test_hardsec_evidence_requires_source_ids() -> None:
    with pytest.raises(ValidationError):
        HardSecEvidence(evidence_ids=[])
