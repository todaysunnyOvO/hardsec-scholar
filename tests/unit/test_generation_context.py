"""Tests for bounded evidence selection and rendering."""

from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation import EvidenceContextBuilder


def _evidence(evidence_id: str, paper_id: str, page: int) -> Evidence:
    return Evidence(
        id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        paper_id=paper_id,
        paper_title=f"Paper {paper_id}",
        section="Evaluation",
        page_start=page,
        page_end=page,
        text=f"Evidence text {evidence_id}",
    )


def test_context_selection_deduplicates_and_caps_each_paper() -> None:
    first = _evidence("E1", "P1", 1)
    ranked = [
        first,
        first,
        _evidence("E2", "P1", 2),
        _evidence("E3", "P2", 3),
        _evidence("E4", "P3", 4),
    ]

    selected = EvidenceContextBuilder(max_evidence=3, max_per_paper=1).select(ranked)

    assert [item.id for item in selected] == ["E1", "E3", "E4"]


def test_context_render_keeps_provenance_and_source_boundaries() -> None:
    rendered = EvidenceContextBuilder.render([_evidence("E1", "P1", 7)])

    assert '<evidence id="E1">' in rendered
    assert "Paper: Paper P1" in rendered
    assert "Section: Evaluation" in rendered
    assert "Pages: 7" in rendered
    assert rendered.endswith("</evidence>")
