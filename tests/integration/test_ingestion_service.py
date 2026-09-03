"""Integration tests for local PDF ingestion."""

from pathlib import Path

import pytest

from hardsec_scholar.config import AppSettings
from hardsec_scholar.domain import PaperStatus
from hardsec_scholar.ingestion.service import DuplicatePaperError, PaperIngestionService
from hardsec_scholar.storage import PaperRepository


def test_ingest_paper_end_to_end(sample_paper_pdf: Path, tmp_path: Path) -> None:
    settings = AppSettings.model_validate(
        {
            "paths": {
                "papers": tmp_path / "papers",
                "chroma": tmp_path / "chroma",
                "evaluations": tmp_path / "evaluations",
                "database": tmp_path / "hardsec.db",
            },
            "chunking": {"chunk_size_tokens": 100, "overlap_tokens": 10},
        }
    )
    repository = PaperRepository(settings.paths.database)
    service = PaperIngestionService(settings, repository)

    result = service.ingest(sample_paper_pdf)

    assert result.paper.status is PaperStatus.PARSED
    assert result.chunk_count >= 3
    assert result.paper.file_path.is_file()
    assert len(repository.get_chunks(result.paper.id)) == result.chunk_count

    with pytest.raises(DuplicatePaperError, match=result.paper.id):
        service.ingest(sample_paper_pdf)
