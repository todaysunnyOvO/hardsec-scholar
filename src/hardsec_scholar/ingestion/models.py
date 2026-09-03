"""Intermediate models produced while parsing papers."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hardsec_scholar.domain import ResearchArea


class TextBlock(BaseModel):
    """Represent one positioned text block on a PDF page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]
    max_font_size: float = Field(gt=0)


class ParsedPage(BaseModel):
    """Represent extracted text and blocks for one physical PDF page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(gt=0)
    text: str
    blocks: tuple[TextBlock, ...] = ()


class ParsedDocument(BaseModel):
    """Represent validated page-aware PDF extraction output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: Path
    content_hash: str = Field(min_length=64, max_length=64)
    pages: tuple[ParsedPage, ...]

    @property
    def page_count(self) -> int:
        """Return the physical page count."""
        return len(self.pages)

    @property
    def text(self) -> str:
        """Join extracted pages for metadata classification."""
        return "\n\n".join(page.text for page in self.pages)


class ExtractedMetadata(BaseModel):
    """Represent heuristically extracted bibliographic metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    authors: tuple[str, ...] = ()
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    abstract: str | None = None
    research_area: ResearchArea = ResearchArea.OTHER
