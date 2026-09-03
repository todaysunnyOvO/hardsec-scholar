"""Deterministic evidence selection and prompt context rendering."""

from collections import defaultdict

from hardsec_scholar.domain import Evidence


class EvidenceContextBuilder:
    """Bound context size while retaining individual source boundaries."""

    def __init__(self, *, max_evidence: int, max_per_paper: int) -> None:
        """Validate and store context limits."""
        if max_evidence <= 0 or max_per_paper <= 0:
            raise ValueError("Context limits must be positive")
        if max_per_paper > max_evidence:
            raise ValueError("max_per_paper must not exceed max_evidence")
        self.max_evidence = max_evidence
        self.max_per_paper = max_per_paper

    def select(self, ranked: list[Evidence]) -> list[Evidence]:
        """Deduplicate ranked evidence and enforce a per-paper cap."""
        selected: list[Evidence] = []
        seen: set[str] = set()
        paper_counts: dict[str, int] = defaultdict(int)
        for item in ranked:
            if item.id in seen or paper_counts[item.paper_id] >= self.max_per_paper:
                continue
            selected.append(item)
            seen.add(item.id)
            paper_counts[item.paper_id] += 1
            if len(selected) >= self.max_evidence:
                break
        return selected

    @staticmethod
    def render(evidence: list[Evidence]) -> str:
        """Render isolated evidence blocks with stable identifiers and provenance."""
        blocks: list[str] = []
        for item in evidence:
            section = item.section or "Unknown section"
            pages = (
                str(item.page_start)
                if item.page_start == item.page_end
                else f"{item.page_start}-{item.page_end}"
            )
            blocks.append(
                "\n".join(
                    [
                        f'<evidence id="{item.id}">',
                        f"Paper: {item.paper_title}",
                        f"Section: {section}",
                        f"Pages: {pages}",
                        item.text,
                        "</evidence>",
                    ]
                )
            )
        return "\n\n".join(blocks)
