"""HTTP request and response schemas for the local application API."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hardsec_scholar.api.history import StoredRun
from hardsec_scholar.domain import PaperMetadata, ResearchArea


class PaperView(BaseModel):
    """Return paper metadata plus its indexed chunk count."""

    model_config = ConfigDict(extra="forbid")

    paper: PaperMetadata
    chunk_count: int = Field(ge=0)


class PaperUploadResponse(PaperView):
    """Report ingestion and optional vector-indexing results."""

    indexed: bool
    indexing_warning: str | None = None


class PaperPatch(BaseModel):
    """Accept only metadata fields that users may correct."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1)
    authors: list[str] | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    research_area: ResearchArea | None = None

    @model_validator(mode="after")
    def require_one_change(self) -> "PaperPatch":
        """Reject empty metadata update requests."""
        if not self.model_fields_set:
            raise ValueError("At least one metadata field must be supplied")
        return self


class MessageRequest(BaseModel):
    """Submit one corpus-only research question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    allow_web_search: bool = False

    @model_validator(mode="after")
    def reject_web_search(self) -> "MessageRequest":
        """Keep the first release within the indexed local corpus."""
        if self.allow_web_search:
            raise ValueError("Web search is not enabled for this local-paper project")
        return self


class MessageResponse(BaseModel):
    """Return the persisted run created for a question."""

    model_config = ConfigDict(extra="forbid")

    run: StoredRun


class ReindexResponse(BaseModel):
    """Report a completed paper vector reindex."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    indexed_chunks: int = Field(gt=0)
