"""Tests for validated application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from hardsec_scholar.config import AppSettings, RuntimeSettings, load_app_settings


def test_load_default_settings() -> None:
    settings = load_app_settings()

    assert settings.retrieval.rerank_top_k == 6
    assert settings.agent.max_retrieval_retries == 2
    assert settings.generation.max_context_evidence == 6
    assert settings.generation.max_evidence_per_paper == 3
    assert settings.paths.papers == Path("data/papers")


def test_reject_invalid_chunk_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap_tokens"):
        AppSettings.model_validate(
            {"chunking": {"chunk_size_tokens": 200, "overlap_tokens": 200}}
        )


def test_reject_per_paper_context_larger_than_total() -> None:
    with pytest.raises(ValidationError, match="max_evidence_per_paper"):
        AppSettings.model_validate(
            {
                "generation": {
                    "max_context_evidence": 2,
                    "max_evidence_per_paper": 3,
                }
            }
        )


def test_runtime_settings_do_not_require_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = RuntimeSettings(_env_file=None)

    assert settings.llm_api_key is None
    assert settings.allow_web_search is False
