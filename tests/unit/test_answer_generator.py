"""Tests for strict structured model prompting."""

from typing import Any

from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation import StructuredAnswerGenerator
from hardsec_scholar.generation.models import AnswerDraft


class _FakeStructuredModel:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        self.messages = messages
        return {
            "answerable": True,
            "claims": [{"text": "A grounded fact.", "evidence_ids": ["E1"]}],
            "missing_evidence": [],
        }

    def with_retry(self, *, stop_after_attempt: int) -> "_FakeStructuredModel":
        assert stop_after_attempt == 3
        return self


class _FlakyStructuredModel(_FakeStructuredModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        if self.calls == 1:
            return None
        return super().invoke(messages)


class _FakeChatModel:
    def __init__(self, structured: _FakeStructuredModel) -> None:
        self.structured = structured
        self.schema: type[AnswerDraft] | None = None
        self.method: str | None = None

    def with_structured_output(
        self, schema: type[AnswerDraft], *, method: str
    ) -> _FakeStructuredModel:
        self.schema = schema
        self.method = method
        return self.structured


def test_generator_binds_schema_and_includes_only_supplied_context() -> None:
    structured = _FakeStructuredModel()
    model = _FakeChatModel(structured)
    generator = StructuredAnswerGenerator(model)  # type: ignore[arg-type]
    evidence = Evidence(
        id="E1",
        chunk_id="C1",
        paper_id="P1",
        paper_title="Paper One",
        section="Methodology",
        page_start=3,
        page_end=3,
        text="The method uses coverage feedback.",
    )

    result = generator.generate("How does it work?", [evidence])

    assert model.schema is AnswerDraft
    assert model.method == "function_calling"
    assert result.claims[0].evidence_ids == ["E1"]
    human_prompt = str(structured.messages[1].content)
    assert "How does it work?" in human_prompt
    assert '<evidence id="E1">' in human_prompt
    assert "Pages: 3" in human_prompt


def test_generator_retries_empty_structured_response() -> None:
    structured = _FlakyStructuredModel()
    model = _FakeChatModel(structured)
    generator = StructuredAnswerGenerator(model)  # type: ignore[arg-type]
    evidence = Evidence(
        id="E1",
        chunk_id="C1",
        paper_id="P1",
        paper_title="Paper One",
        page_start=3,
        page_end=3,
        text="A grounded fact.",
    )

    result = generator.generate("What is grounded?", [evidence])

    assert result.answerable is True
    assert structured.calls == 2
