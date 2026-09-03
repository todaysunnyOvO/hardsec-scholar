"""Create section-aware, page-preserving chunks from parsed papers."""

import re
from dataclasses import dataclass

import tiktoken

from hardsec_scholar.domain import PaperChunk
from hardsec_scholar.ingestion.models import ParsedDocument

SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "threat model": "Threat Model",
    "attack model": "Attack Model",
    "method": "Method",
    "methodology": "Methodology",
    "implementation": "Implementation",
    "experimental setup": "Experimental Setup",
    "evaluation": "Evaluation",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
}
NUMBERING_PATTERN = re.compile(
    r"^(?:(?:[IVXLCDM]+|\d+(?:\.\d+)*)[.\s:\-]+)", re.IGNORECASE
)
INLINE_HEADING_PATTERN = re.compile(
    r"^(abstract|introduction|background|related work|threat model|attack model|"
    r"methodology|method|implementation|experimental setup|evaluation|results|"
    r"discussion|limitations|conclusions?|references)\s*[—–:\-]\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _TextUnit:
    text: str
    section: str
    page_number: int
    token_count: int


def normalize_section_heading(text: str) -> str | None:
    """Normalize a conservative set of common scientific section headings."""
    collapsed = " ".join(text.split()).strip().rstrip(".:—–-")
    if not collapsed or len(collapsed) > 80:
        return None
    without_number = NUMBERING_PATTERN.sub("", collapsed).strip().lower()
    return SECTION_ALIASES.get(without_number)


def _split_long_text(
    text: str,
    *,
    section: str,
    page_number: int,
    encoding: tiktoken.Encoding,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[_TextUnit]:
    token_ids = encoding.encode(text)
    if len(token_ids) <= chunk_size_tokens:
        return [_TextUnit(text, section, page_number, len(token_ids))]

    units: list[_TextUnit] = []
    step = chunk_size_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_size_tokens]
        if not window:
            break
        units.append(
            _TextUnit(
                text=encoding.decode(window).strip(),
                section=section,
                page_number=page_number,
                token_count=len(window),
            )
        )
        if start + chunk_size_tokens >= len(token_ids):
            break
    return units


def _extract_units(
    document: ParsedDocument,
    *,
    encoding: tiktoken.Encoding,
    chunk_size_tokens: int,
    overlap_tokens: int,
    exclude_references: bool,
) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    current_section = "Front Matter"
    for page in document.pages:
        for block in page.blocks:
            text = block.text.strip()
            inline_match = INLINE_HEADING_PATTERN.match(text)
            if inline_match:
                current_section = SECTION_ALIASES[inline_match.group(1).lower()]
                text = inline_match.group(2).strip()
            else:
                heading = normalize_section_heading(text)
                if heading:
                    current_section = heading
                    continue

            if exclude_references and current_section == "References":
                continue
            if not text:
                continue
            units.extend(
                _split_long_text(
                    text,
                    section=current_section,
                    page_number=page.page_number,
                    encoding=encoding,
                    chunk_size_tokens=chunk_size_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
    return units


def chunk_document(
    document: ParsedDocument,
    *,
    paper_id: str,
    title: str,
    chunk_size_tokens: int = 700,
    overlap_tokens: int = 100,
    exclude_references: bool = True,
) -> list[PaperChunk]:
    """Pack parsed blocks into section-aware chunks with bounded overlap."""
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    encoding = tiktoken.get_encoding("cl100k_base")
    units = _extract_units(
        document,
        encoding=encoding,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        exclude_references=exclude_references,
    )

    packed: list[list[_TextUnit]] = []
    current: list[_TextUnit] = []
    current_tokens = 0
    current_section: str | None = None

    for unit in units:
        section_changed = (
            current_section is not None and unit.section != current_section
        )
        exceeds_limit = (
            current and current_tokens + unit.token_count > chunk_size_tokens
        )
        if section_changed or exceeds_limit:
            packed.append(current)
            carry: list[_TextUnit] = []
            carry_tokens = 0
            if not section_changed and overlap_tokens:
                for previous in reversed(current):
                    if carry_tokens + previous.token_count > overlap_tokens:
                        break
                    carry.insert(0, previous)
                    carry_tokens += previous.token_count
            current = carry
            current_tokens = carry_tokens

        current.append(unit)
        current_tokens += unit.token_count
        current_section = unit.section

    if current:
        packed.append(current)

    chunks: list[PaperChunk] = []
    for index, group in enumerate(packed):
        if not group:
            continue
        chunks.append(
            PaperChunk(
                id=f"{paper_id}_chunk_{index:04d}",
                paper_id=paper_id,
                title=title,
                section=group[0].section,
                page_start=min(unit.page_number for unit in group),
                page_end=max(unit.page_number for unit in group),
                chunk_index=index,
                text="\n\n".join(unit.text for unit in group),
            )
        )
    return chunks
