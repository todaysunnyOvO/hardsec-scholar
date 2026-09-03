"""Paper ingestion pipeline."""

from hardsec_scholar.ingestion.service import (
    DuplicatePaperError,
    IngestionResult,
    PaperIngestionService,
)

__all__ = ["DuplicatePaperError", "IngestionResult", "PaperIngestionService"]
