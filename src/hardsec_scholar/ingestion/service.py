"""Orchestrate PDF parsing, chunking, local copying, and persistence."""

import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hardsec_scholar.config import AppSettings
from hardsec_scholar.domain import PaperMetadata, PaperStatus
from hardsec_scholar.ingestion.chunker import chunk_document
from hardsec_scholar.ingestion.metadata import extract_metadata
from hardsec_scholar.ingestion.parser import parse_pdf
from hardsec_scholar.storage import PaperRepository


class DuplicatePaperError(ValueError):
    """Indicate that the exact PDF content already exists in the corpus."""

    def __init__(self, paper_id: str) -> None:
        """Record the stable ID of the previously ingested paper."""
        super().__init__(f"Paper already exists: {paper_id}")
        self.paper_id = paper_id


class IngestionResult(BaseModel):
    """Summarize one completed local ingestion operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper: PaperMetadata
    chunk_count: int = Field(gt=0)


class PaperIngestionService:
    """Coordinate deterministic paper ingestion without external model calls."""

    def __init__(self, settings: AppSettings, repository: PaperRepository) -> None:
        """Initialize the service with validated settings and local persistence."""
        self.settings = settings
        self.repository = repository

    def ingest(self, source_path: Path | str) -> IngestionResult:
        """Parse, copy, chunk, and persist one unique PDF."""
        self.repository.initialize()
        document = parse_pdf(
            source_path,
            max_file_size_mb=self.settings.ingestion.max_file_size_mb,
        )
        duplicate = self.repository.find_by_hash(document.content_hash)
        if duplicate:
            raise DuplicatePaperError(duplicate.id)

        metadata = extract_metadata(document)
        paper_id = f"paper_{document.content_hash[:16]}"
        papers_path = self.settings.paths.papers
        papers_path.mkdir(parents=True, exist_ok=True)
        stored_path = papers_path / f"{paper_id}.pdf"

        chunks = chunk_document(
            document,
            paper_id=paper_id,
            title=metadata.title,
            chunk_size_tokens=self.settings.chunking.chunk_size_tokens,
            overlap_tokens=self.settings.chunking.overlap_tokens,
            exclude_references=self.settings.ingestion.exclude_references,
        )
        if not chunks:
            raise ValueError("PDF produced no indexable chunks")

        source = document.source_path.resolve()
        target = stored_path.resolve()
        if source != target:
            shutil.copy2(source, target)

        paper = PaperMetadata(
            id=paper_id,
            content_hash=document.content_hash,
            title=metadata.title,
            authors=list(metadata.authors),
            year=metadata.year,
            doi=metadata.doi,
            research_area=metadata.research_area,
            file_path=stored_path,
            page_count=document.page_count,
            status=PaperStatus.PARSED,
        )
        try:
            self.repository.save_paper(paper, chunks)
        except Exception:
            if source != target and target.is_file():
                target.unlink()
            raise
        return IngestionResult(paper=paper, chunk_count=len(chunks))
