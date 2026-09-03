"""LangChain structured-output adapter for evidence-only answer generation."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from hardsec_scholar.domain import Evidence
from hardsec_scholar.generation.context import EvidenceContextBuilder
from hardsec_scholar.generation.models import AnswerDraft

SYSTEM_PROMPT = """You answer questions about hardware-security research papers.
Use only the supplied evidence blocks. Do not use background knowledge or infer facts
that the evidence does not support. Return short, atomic claims. Every claim must cite
one or more exact evidence IDs from the context. If the evidence is insufficient,
set answerable to false, return no claims, and describe the missing evidence.
Answer in the same language as the user's question."""


class StructuredAnswerGenerator:
    """Invoke a chat model with a strict Pydantic output schema."""

    def __init__(self, model: BaseChatModel) -> None:
        """Bind the supplied LangChain model to the answer schema."""
        self.model = model.with_structured_output(
            AnswerDraft, method="function_calling"
        ).with_retry(stop_after_attempt=3)

    def generate(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        """Generate a schema-validated answer draft from bounded evidence."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        if not evidence:
            raise ValueError("Evidence must not be empty")
        context = EvidenceContextBuilder.render(evidence)
        validation_error: Exception | None = None
        for _ in range(3):
            result = self.model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Question:\n{question}\n\nEvidence:\n{context}"
                    ),
                ]
            )
            try:
                return AnswerDraft.model_validate(result)
            except Exception as exc:
                validation_error = exc
        if validation_error is None:
            raise RuntimeError("Structured answer validation did not run")
        raise validation_error
