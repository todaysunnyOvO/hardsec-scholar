"""Integration tests for bounded Agentic RAG graph transitions."""

from collections import deque

from hardsec_scholar.agent import (
    AgenticRAGWorkflow,
    EvidenceGrade,
    QueryRewrite,
    QuestionPlan,
    SemanticCitationCheck,
)
from hardsec_scholar.config import AgentSettings, GenerationSettings
from hardsec_scholar.domain import Evidence, QuestionType
from hardsec_scholar.generation import AnswerDraft, AnswerStatus, ClaimDraft


class _QueryRetriever:
    def __init__(self, results: dict[str, list[Evidence]]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        self.calls.append(query)
        return self.results.get(query, [])


class _FixedGenerator:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls = 0

    def generate(self, question: str, evidence: list[Evidence]) -> AnswerDraft:
        self.calls += 1
        return self.draft


class _ScriptedReasoner:
    def __init__(
        self,
        *,
        grades: list[EvidenceGrade] | None = None,
        rewrites: list[QueryRewrite] | None = None,
        checks: list[SemanticCitationCheck] | None = None,
        repaired: AnswerDraft | None = None,
        plan: QuestionPlan | None = None,
    ) -> None:
        self.grades = deque(grades or [])
        self.rewrites = deque(rewrites or [])
        self.checks = deque(checks or [])
        self.repaired = repaired
        self.plan = plan or QuestionPlan(question_type=QuestionType.FACT)
        self.repair_calls = 0

    def classify_and_plan(self, question: str) -> QuestionPlan:
        return self.plan

    def grade_evidence(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        selected_paper_ids: list[str],
        requires_comparison: bool,
    ) -> EvidenceGrade:
        return self.grades.popleft()

    def rewrite_query(
        self,
        question: str,
        *,
        previous_queries: list[str],
        missing_aspects: list[str],
    ) -> QueryRewrite:
        return self.rewrites.popleft()

    def verify_citations(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
    ) -> SemanticCitationCheck:
        return self.checks.popleft()

    def repair_answer(
        self,
        question: str,
        draft: AnswerDraft,
        evidence: list[Evidence],
        *,
        verification_errors: list[str],
    ) -> AnswerDraft:
        self.repair_calls += 1
        assert self.repaired is not None
        return self.repaired


def _evidence(evidence_id: str = "E1") -> Evidence:
    return Evidence(
        id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        paper_id="P1",
        paper_title="Hardware Fuzzing",
        section="Evaluation",
        page_start=4,
        page_end=4,
        text="Coverage feedback guides hardware fuzzing mutation selection.",
    )


def _draft(text: str = "Coverage feedback guides mutations.") -> AnswerDraft:
    return AnswerDraft(
        answerable=True,
        claims=[ClaimDraft(text=text, evidence_ids=["E1"])],
    )


def _workflow(
    retriever: _QueryRetriever,
    reasoner: _ScriptedReasoner,
    *,
    generator: _FixedGenerator | None = None,
    retries: int = 2,
    repairs: int = 1,
    terminology: dict[str, tuple[str, ...]] | None = None,
) -> tuple[AgenticRAGWorkflow, _FixedGenerator]:
    answer_generator = generator or _FixedGenerator(_draft())
    workflow = AgenticRAGWorkflow(
        retriever=retriever,
        generator=answer_generator,
        reasoner=reasoner,
        terminology=terminology or {},
        agent_settings=AgentSettings(
            max_retrieval_retries=retries,
            max_answer_repairs=repairs,
        ),
        generation_settings=GenerationSettings(),
    )
    return workflow, answer_generator


def test_agent_completes_with_semantically_verified_citation() -> None:
    question = "How does RTL fuzzing use coverage?"
    retriever = _QueryRetriever(
        {
            question: [_evidence()],
            f"{question} register-transfer level": [_evidence()],
        }
    )
    reasoner = _ScriptedReasoner(
        grades=[EvidenceGrade(sufficient=True, selected_evidence_ids=["E1"])],
        checks=[SemanticCitationCheck(supported=True)],
    )
    workflow, _ = _workflow(
        retriever,
        reasoner,
        terminology={"rtl": ("register-transfer level",)},
    )

    run = workflow.run(question, paper_ids=["P1"])

    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.answer.citations[0].evidence_id == "E1"
    assert run.retrieval_retries == 0
    assert run.search_queries == [question, f"{question} register-transfer level"]
    assert [event.sequence for event in run.trace_events] == list(
        range(1, len(run.trace_events) + 1)
    )
    assert run.trace_events[-1].event == "completed"


def test_agent_rewrites_query_then_answers() -> None:
    question = "What latency was measured?"
    rewritten = "measured latency evaluation cycles"
    retriever = _QueryRetriever({rewritten: [_evidence()]})
    reasoner = _ScriptedReasoner(
        grades=[EvidenceGrade(sufficient=True, selected_evidence_ids=["E1"])],
        rewrites=[QueryRewrite(query=rewritten, reason="Find the missing metric.")],
        checks=[SemanticCitationCheck(supported=True)],
    )
    workflow, _ = _workflow(retriever, reasoner)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.retrieval_retries == 1
    assert run.search_queries == [question, rewritten]
    assert run.rewrite_reasons == ["Find the missing metric."]
    assert "query_rewritten" in [event.event for event in run.trace_events]


def test_agent_refuses_after_retrieval_retry_limit() -> None:
    question = "What is the silicon area overhead?"
    retriever = _QueryRetriever({})
    reasoner = _ScriptedReasoner(
        rewrites=[
            QueryRewrite(query="silicon area evaluation", reason="Find area results."),
            QueryRewrite(
                query="gate count overhead", reason="Try another area metric."
            ),
        ]
    )
    workflow, generator = _workflow(retriever, reasoner, retries=2)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ABSTAINED
    assert run.retrieval_retries == 2
    assert generator.calls == 0
    assert len(retriever.calls) == 3
    assert run.answer.missing_evidence


def test_agent_tries_verified_generation_after_conservative_grade_limit() -> None:
    question = "How does the mechanism work?"
    rewrite_one = "mechanism implementation details"
    rewrite_two = "mechanism evidence"
    evidence = _evidence()
    retriever = _QueryRetriever(
        {
            question: [evidence],
            rewrite_one: [evidence],
            rewrite_two: [evidence],
        }
    )
    insufficient = EvidenceGrade(
        sufficient=False,
        selected_evidence_ids=["E1"],
        missing_aspects=["The grader requests more detail."],
    )
    reasoner = _ScriptedReasoner(
        grades=[insufficient, insufficient, insufficient],
        rewrites=[
            QueryRewrite(query=rewrite_one, reason="Find implementation details."),
            QueryRewrite(query=rewrite_two, reason="Try a focused evidence query."),
        ],
        checks=[SemanticCitationCheck(supported=True)],
    )
    workflow, generator = _workflow(retriever, reasoner, retries=2)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.retrieval_retries == 2
    assert generator.calls == 1


def test_agent_rejects_duplicate_rewrite_without_looping() -> None:
    question = "What is the overhead?"
    retriever = _QueryRetriever({})
    reasoner = _ScriptedReasoner(
        rewrites=[QueryRewrite(query=question, reason="Retry the question.")]
    )
    workflow, generator = _workflow(retriever, reasoner)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ABSTAINED
    assert run.retrieval_retries == 0
    assert generator.calls == 0
    assert "repeated" in run.answer.verification_errors[0]
    assert retriever.calls == [question]


def test_agent_repairs_failed_semantic_citation_once() -> None:
    question = "Does coverage guarantee bug discovery?"
    retriever = _QueryRetriever({question: [_evidence()]})
    repaired = _draft("Coverage feedback guides mutation selection.")
    reasoner = _ScriptedReasoner(
        grades=[EvidenceGrade(sufficient=True, selected_evidence_ids=["E1"])],
        checks=[
            SemanticCitationCheck(
                supported=False,
                unsupported_claim_indices=[1],
                errors=["Evidence does not support a guarantee."],
            ),
            SemanticCitationCheck(supported=True),
        ],
        repaired=repaired,
    )
    generator = _FixedGenerator(_draft("Coverage guarantees bug discovery."))
    workflow, _ = _workflow(
        retriever,
        reasoner,
        generator=generator,
        repairs=1,
    )

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ANSWERED
    assert "guides mutation selection" in run.answer.answer
    assert "guarantees" not in run.answer.answer
    assert run.answer_repairs == 1
    assert reasoner.repair_calls == 1


def test_agent_refuses_grader_fabricated_evidence_id() -> None:
    question = "What does coverage guide?"
    retriever = _QueryRetriever({question: [_evidence()]})
    reasoner = _ScriptedReasoner(
        grades=[EvidenceGrade(sufficient=True, selected_evidence_ids=["E999"])]
    )
    workflow, generator = _workflow(retriever, reasoner, retries=0)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ABSTAINED
    assert generator.calls == 0
    assert "unavailable evidence IDs" in run.answer.missing_evidence[0]


def test_agent_refuses_when_bounded_answer_repair_still_fails() -> None:
    question = "Does coverage guarantee finding every bug?"
    retriever = _QueryRetriever({question: [_evidence()]})
    unsupported = SemanticCitationCheck(
        supported=False,
        unsupported_claim_indices=[1],
        errors=["The evidence does not establish a guarantee."],
    )
    reasoner = _ScriptedReasoner(
        grades=[EvidenceGrade(sufficient=True, selected_evidence_ids=["E1"])],
        checks=[unsupported, unsupported],
        repaired=_draft("Coverage guarantees finding every bug."),
    )
    workflow, _ = _workflow(retriever, reasoner, repairs=1)

    run = workflow.run(question)

    assert run.answer.status is AnswerStatus.ABSTAINED
    assert run.answer_repairs == 1
    assert reasoner.repair_calls == 1
    assert run.answer.verification_errors == [
        "The evidence does not establish a guarantee."
    ]
