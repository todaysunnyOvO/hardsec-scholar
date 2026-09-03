"""Tests for resilient structured Agent decisions."""

from typing import Any

from hardsec_scholar.agent.models import QuestionPlan
from hardsec_scholar.agent.reasoner import StructuredAgentReasoner
from hardsec_scholar.domain import QuestionType


class _StructuredModel:
    def __init__(self, schema: type[Any], *, always_empty: bool = False) -> None:
        self.schema = schema
        self.calls = 0
        self.always_empty = always_empty

    def with_retry(self, *, stop_after_attempt: int) -> "_StructuredModel":
        assert stop_after_attempt == 3
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        if self.schema is QuestionPlan:
            if self.always_empty or self.calls == 1:
                return None
            return {
                "question_type": QuestionType.FACT,
                "sub_questions": [],
                "preferred_sections": ["Introduction"],
                "requires_comparison": False,
            }
        raise AssertionError("Unexpected structured model invocation")


class _ChatModel:
    def __init__(self, *, always_empty_plan: bool = False) -> None:
        self.models: dict[type[Any], _StructuredModel] = {}
        self.always_empty_plan = always_empty_plan

    def with_structured_output(
        self, schema: type[Any], *, method: str
    ) -> _StructuredModel:
        assert method == "function_calling"
        model = _StructuredModel(
            schema,
            always_empty=self.always_empty_plan and schema is QuestionPlan,
        )
        self.models[schema] = model
        return model


def test_reasoner_retries_empty_structured_response() -> None:
    model = _ChatModel()
    reasoner = StructuredAgentReasoner(model)  # type: ignore[arg-type]

    plan = reasoner.classify_and_plan("What does the paper contribute?")

    assert plan.question_type is QuestionType.FACT
    assert model.models[QuestionPlan].calls == 2


def test_reasoner_falls_back_after_repeated_empty_plans() -> None:
    model = _ChatModel(always_empty_plan=True)
    reasoner = StructuredAgentReasoner(model)  # type: ignore[arg-type]

    plan = reasoner.classify_and_plan("How do the two papers differ?")

    assert plan.question_type is QuestionType.COMPARISON
    assert plan.requires_comparison is True
    assert model.models[QuestionPlan].calls == 3
