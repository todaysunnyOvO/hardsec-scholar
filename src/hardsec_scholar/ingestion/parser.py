"""Validate and extract page-aware content from text-based PDFs."""

import hashlib
from pathlib import Path
from typing import Any

import pymupdf

from hardsec_scholar.ingestion.models import ParsedDocument, ParsedPage, TextBlock


class PdfValidationError(ValueError):
    """Indicate that an input file cannot enter the ingestion pipeline."""


class ScannedPdfError(PdfValidationError):
    """Indicate that a PDF has too little extractable text and likely needs OCR."""


def calculate_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a stable SHA-256 content hash without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(path: Path | str, *, max_file_size_mb: int = 50) -> Path:
    """Validate existence, extension, size, and basic PDF readability."""
    pdf_path = Path(path).resolve()
    if not pdf_path.is_file():
        raise PdfValidationError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PdfValidationError("Only PDF files are supported")
    if pdf_path.stat().st_size == 0:
        raise PdfValidationError("PDF file is empty")
    if pdf_path.stat().st_size > max_file_size_mb * 1024 * 1024:
        raise PdfValidationError(f"PDF exceeds the {max_file_size_mb} MB size limit")
    return pdf_path


def _extract_blocks(page: pymupdf.Page) -> tuple[TextBlock, ...]:
    page_dict: dict[str, Any] = page.get_text("dict", sort=True)
    extracted: list[TextBlock] = []
    for raw_block in page_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue

        line_texts: list[str] = []
        font_sizes: list[float] = []
        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if text:
                line_texts.append(text)
            font_sizes.extend(
                float(span.get("size", 0)) for span in spans if span.get("size")
            )

        block_text = "\n".join(line_texts).strip()
        bbox = raw_block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if not block_text or not font_sizes or len(bbox) != 4:
            continue
        extracted.append(
            TextBlock(
                text=block_text,
                bbox=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
                max_font_size=max(font_sizes),
            )
        )
    return tuple(extracted)


def parse_pdf(
    path: Path | str,
    *,
    max_file_size_mb: int = 50,
    minimum_text_characters: int = 50,
) -> ParsedDocument:
    """Extract text blocks while preserving physical page numbers."""
    pdf_path = validate_pdf(path, max_file_size_mb=max_file_size_mb)
    pages: list[ParsedPage] = []

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise PdfValidationError("Password-protected PDFs are not supported")
            if document.page_count == 0:
                raise PdfValidationError("PDF contains no pages")

            for page_index, page in enumerate(document):
                blocks = _extract_blocks(page)
                page_text = "\n\n".join(block.text for block in blocks)
                pages.append(
                    ParsedPage(
                        page_number=page_index + 1,
                        text=page_text,
                        blocks=blocks,
                    )
                )
    except PdfValidationError:
        raise
    except Exception as exc:
        raise PdfValidationError(f"Unable to parse PDF: {pdf_path.name}") from exc

    extracted_characters = sum(len(page.text.strip()) for page in pages)
    if extracted_characters < minimum_text_characters:
        raise ScannedPdfError(
            "PDF contains too little extractable text; scanned documents require OCR"
        )

    return ParsedDocument(
        source_path=pdf_path,
        content_hash=calculate_sha256(pdf_path),
        pages=tuple(pages),
    )
