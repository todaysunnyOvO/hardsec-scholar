"""Tests for transactional SQLite paper persistence."""

from pathlib import Path

from hardsec_scholar.domain import PaperChunk, PaperMetadata, PaperStatus
from hardsec_scholar.storage import PaperRepository


def test_repository_round_trip_and_cascade_delete(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "papers.db")
    repository.initialize()
    paper = PaperMetadata(
        id="paper-1",
        content_hash="a" * 64,
        title="Paper",
        file_path=Path("paper.pdf"),
        page_count=3,
        status=PaperStatus.PARSED,
    )
    chunk = PaperChunk(
        id="paper-1_chunk_0000",
        paper_id="paper-1",
        title="Paper",
        section="Introduction",
        page_start=1,
        page_end=1,
        chunk_index=0,
        text="Chunk text",
    )

    repository.save_paper(paper, [chunk])

    assert repository.get_paper("paper-1") == paper
    assert repository.find_by_hash("a" * 64) == paper
    assert repository.get_chunks("paper-1") == [chunk]
    assert repository.delete_paper("paper-1") is True
    assert repository.get_chunks("paper-1") == []
