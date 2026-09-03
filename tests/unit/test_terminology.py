"""Tests for deterministic terminology expansion."""

import pytest

from hardsec_scholar.domain.terminology import expand_query, load_terminology


def test_load_and_expand_known_acronym() -> None:
    terminology = load_terminology()

    variants = expand_query("Which SCA needs fewer traces?", terminology)

    assert variants[0] == "Which SCA needs fewer traces?"
    assert "Which SCA needs fewer traces? side-channel attack" in variants
    assert "Which SCA needs fewer traces? side channel analysis" in variants


def test_unknown_term_keeps_original_query_only() -> None:
    terminology = load_terminology()

    assert expand_query("Explain cache coherence", terminology) == [
        "Explain cache coherence"
    ]


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="query"):
        expand_query("  ", {})
