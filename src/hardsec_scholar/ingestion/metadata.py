"""Extract bibliographic and hardware-security metadata from parsed papers."""

import re

from hardsec_scholar.domain import ResearchArea
from hardsec_scholar.ingestion.models import (
    ExtractedMetadata,
    ParsedDocument,
    TextBlock,
)

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
ABSTRACT_PATTERN = re.compile(
    r"\babstract\b\s*[—–:\-]?\s*(.+?)(?=\n\s*(?:I\.?\s+)?introduction\b)",
    re.IGNORECASE | re.DOTALL,
)

AREA_KEYWORDS: dict[ResearchArea, tuple[str, ...]] = {
    ResearchArea.SIDE_CHANNEL_ATTACK: (
        "side-channel",
        "side channel",
        "power analysis",
        "electromagnetic leakage",
        "cache attack",
        "timing attack",
    ),
    ResearchArea.ARCHITECTURAL_SECURITY: (
        "microarchitectural",
        "architectural security",
        "speculative execution",
        "transient execution",
        "rowhammer",
        "trusted execution environment",
    ),
    ResearchArea.HARDWARE_FUZZING: (
        "hardware fuzzing",
        "hardware fuzzer",
        "rtl fuzzing",
        "processor fuzzing",
        "coverage-guided fuzzing",
        "fuzz testing",
    ),
}


def classify_research_area(text: str) -> ResearchArea:
    """Classify a paper with transparent keyword scoring."""
    lowered = text.lower()
    scores = {
        area: sum(lowered.count(keyword) for keyword in keywords)
        for area, keywords in AREA_KEYWORDS.items()
    }
    best_area, best_score = max(scores.items(), key=lambda item: item[1])
    return best_area if best_score > 0 else ResearchArea.OTHER


def _is_title_candidate(block: TextBlock) -> bool:
    normalized = " ".join(block.text.split())
    if len(normalized) < 12 or len(normalized) > 350:
        return False
    lowered = normalized.lower()
    return not lowered.startswith(("arxiv:", "proceedings of", "ieee "))


def _extract_title(first_page_blocks: tuple[TextBlock, ...]) -> tuple[str, int]:
    candidates = [
        (index, block)
        for index, block in enumerate(first_page_blocks)
        if _is_title_candidate(block)
    ]
    if not candidates:
        return "Untitled paper", -1
    title_index, title_block = max(
        candidates,
        key=lambda item: (item[1].max_font_size, -item[1].bbox[1]),
    )
    return " ".join(title_block.text.split()), title_index


def _extract_authors(
    first_page_blocks: tuple[TextBlock, ...], title_index: int
) -> tuple[str, ...]:
    if title_index < 0:
        return ()
    author_lines: list[str] = []
    for block in first_page_blocks[title_index + 1 :]:
        normalized = " ".join(block.text.split())
        lowered = normalized.lower()
        if lowered.startswith("abstract"):
            break
        if any(
            marker in lowered
            for marker in ("university", "department", "institute", "@")
        ):
            continue
        if len(normalized) <= 250:
            author_lines.append(normalized)
        if len(author_lines) == 2:
            break

    joined = ", ".join(author_lines)
    names = re.split(r"\s*(?:,|\band\b|;)\s*", joined)
    return tuple(name for name in names if 2 <= len(name) <= 100)


def extract_metadata(document: ParsedDocument) -> ExtractedMetadata:
    """Extract best-effort metadata while keeping every field user-editable."""
    first_page = document.pages[0]
    title, title_index = _extract_title(first_page.blocks)
    authors = _extract_authors(first_page.blocks, title_index)
    front_matter = "\n".join(page.text for page in document.pages[:2])

    year_matches = [
        int(match.group(0)) for match in YEAR_PATTERN.finditer(front_matter)
    ]
    year = max(year_matches) if year_matches else None
    doi_match = DOI_PATTERN.search(front_matter)
    abstract_match = ABSTRACT_PATTERN.search(front_matter)

    return ExtractedMetadata(
        title=title,
        authors=authors,
        year=year,
        doi=doi_match.group(0).rstrip(".,") if doi_match else None,
        abstract=" ".join(abstract_match.group(1).split()) if abstract_match else None,
        research_area=classify_research_area(document.text),
    )
