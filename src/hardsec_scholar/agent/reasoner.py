"""LangChain structured-output implementation of Agentic RAG decisions."""

from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from hardsec_scholar.agent.models import (
    EvidenceGrade,
    QueryRewrite,
    QuestionPlan,
    SemanticCitationCheck,
)
from hardsec_scholar.domain import Evidence, QuestionType
from hardsec_scholar.generation import AnswerDraft, EvidenceContextBuilder

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredAgentReasoner:
    """Use one chat model for schema-constrained workflow decisions."""

    def __init__(self, model: BaseChatModel) -> None:
        """Bind separate structured schemas for each reasoning responsibility."""
        self.plan_model = model.with_structured_output(
            QuestionPlan, method="function_calling"
        ).with_retry(
            stop_after_attempt=3
        )
        self.grade_model = model.with_structured_output(
            EvidenceGrade, method="function_calling"
        ).with_retry(
            stop_after_attempt=3
        )
        self.rewrite_model = model.with_structured_output(
            QueryRewrite, method="function_calling"
        ).with_retry(
            stop_after_attempt=3
        )
        self.verify_model = model.with_structured_output(
            SemanticCitationCheck, method="function_calling"
        ).with_retry(stop_after_attempt=3)
        self.repair_model = model.with_structured_output(
            AnswerDraft, method="function_calling"
        ).with_retry(stop_after_attempt=3)

    @staticmethod
    def _invoke(
        model: Any,
        system: str,
        human: str,
        schema: type[StructuredModel],
    ) -> StructuredModel:
        validation_error: Exception | None = None
        for _ in range(3):
            result = model.invoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
            try:
                return schema.model_validate(result)
            except Exception as exc:
                validation_error = exc
        if validation_error is None:
            raise RuntimeError("Structured model validation did not run")
        raise validation_error

    def classify_and_plan(self, question: str) -> QuestionPlan:
        """Classify question type, sub-questions, and useful paper sections."""
        try:
            return self._invoke(
                self.plan_model,
                (
                    "Classify a hardware-security paper question and make a concise "
                    "retrieval plan. Preserve the original scope. Split only comparison "
                    "questions and name likely paper sections."
                ),
                question,
                QuestionPlan,
            )
        except ValidationError:
            return self._fallback_plan(question)

    @staticmethod
    def _fallback_plan(question: str) -> QuestionPlan:
        """Classify deterministically when a provider returns no tool arguments."""
        normalized = question.casefold()
        comparison = any(
            phrase in normalized
            for phrase in ["compare", "comparison", "differ", "difference", "two papers"]
        )
        if comparison:
            question_type = QuestionType.COMPARISON
        elif any(term in normalized for term in ["threat model", "attacker", "adversary"]):
            question_type = QuestionType.THREAT_MODEL
        elif any(
            term in normalized
            for term in [
                "experiment",
                "evaluation",
                "accuracy",
                "runtime",
                "throughput",
                "bit rate",
            ]
        ):
            question_type = QuestionType.EXPERIMENT
        elif any(term in normalized for term in ["how does", "mechanism", "works"]):
            question_type = QuestionType.MECHANISM
        elif any(term in normalized for term in ["limitation", "drawback"]):
            question_type = QuestionType.LIMITATION
        else:
            question_type = QuestionType.FACT
        return QuestionPlan(
            question_type=question_type,
            sub_questions=[],
            preferred_sections=[],
            requires_comparison=comparison,
        )

    def grade_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        selected_paper_ids: list[str],
        requires_comparison: bool,
    ) -> EvidenceGrade:
        """Assess relevance, coverage, and multi-paper comparison completeness."""
        context = EvidenceContextBuilder.render(evidence)
        candidate_paper_ids = sorted({item.paper_id for item in evidence})
        return self._invoke(
            self.grade_model,
            (
                "Grade only the supplied paper evidence. Select exact evidence IDs. "
                "Sufficient means the evidence directly covers the requested facts. "
                "For a comparison, separate evidence from each paper is sufficient: "
                "the sources do not need to contain an existing side-by-side comparison, "
                "because the answer generator will synthesize it. When selected target "
                "paper IDs are provided, require coverage of each one. Otherwise, require "
                "relevant evidence from at least two candidate papers."
            ),
            (
                f"Question: {question}\n"
                f"Selected paper IDs: {selected_paper_ids}\n"
                f"Candidate evidence paper IDs: {candidate_paper_ids}\n"
                f"Comparison required: {requires_comparison}\n\n{context}"
            ),
            EvidenceGrade,
        )

    def rewrite_query(
        self,
        question: str,
        *,
        previous_queries: list[str],
        missing_aspects: list[str],
    ) -> QueryRewrite:
        """Rewrite toward named evidence gaps without dropping original constraints."""
        return self._invoke(
            self.rewrite_model,
            (
                "Rewrite the retrieval query to target the missing evidence. Preserve "
                "the original entities and constraints. The new query must not repeat "
                "a previous query. Do not answer the question."
            ),
            (
                f"Original question: {question}\n"
                f"Previous queries: {previous_queries}\n"
                f"Missing evidence: {missing_aspects}"
            ),
            QueryRewrite,
        )

    def verify_citations(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
    ) -> SemanticCitationCheck:
        """Check semantic support without relying on outside knowledge."""
        return self._invoke(
            self.verify_model,
            (
                "Verify whether every claim is directly supported by its cited evidence. "
                "Use only the supplied evidence. Report unsupported claim indices using "
                "one-based numbering."
            ),
            (
                f"Question: {question}\n"
                f"Claims: {draft.model_dump()}\n\n"
                f"Evidence:\n{EvidenceContextBuilder.render(evidence)}"
            ),
            SemanticCitationCheck,
        )

    def repair_answer(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
        *,
        verification_errors: list[str],
    ) -> AnswerDraft:
        """Remove or correct unsupported claims using the same evidence set."""
        return self._invoke(
            self.repair_model,
            (
                "Repair the answer using only the supplied evidence. Remove unsupported "
                "claims, correct overstatements, and cite only exact evidence IDs shown. "
                "If no supported answer remains, return an unanswerable draft."
            ),
            (
                f"Question: {question}\n"
                f"Draft: {draft.model_dump()}\n"
                f"Verification errors: {verification_errors}\n\n"
                f"Evidence:\n{EvidenceContextBuilder.render(evidence)}"
            ),
            AnswerDraft,
        )
