"""Bounded LangGraph workflow for evidence-seeking and answer repair."""

from collections.abc import Iterator
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from hardsec_scholar.agent.interfaces import AgentReasoner
from hardsec_scholar.agent.models import (
    AgentRun,
    EvidenceGrade,
    ResearchState,
    TraceEvent,
)
from hardsec_scholar.config import AgentSettings, GenerationSettings
from hardsec_scholar.domain.terminology import Terminology, expand_query
from hardsec_scholar.generation import (
    AnswerGenerator,
    AnswerStatus,
    CitationVerifier,
    EvidenceContextBuilder,
    EvidenceRetriever,
    GroundedAnswer,
)
from hardsec_scholar.generation.service import render_claims


class AgenticRAGWorkflow:
    """Run bounded retrieval-rewrite and answer-repair loops with LangGraph."""

    def __init__(
        self,
        *,
        retriever: EvidenceRetriever,
        generator: AnswerGenerator,
        reasoner: AgentReasoner,
        terminology: Terminology,
        agent_settings: AgentSettings,
        generation_settings: GenerationSettings,
        citation_verifier: CitationVerifier | None = None,
    ) -> None:
        """Store dependencies and compile the reusable state graph."""
        self.retriever = retriever
        self.generator = generator
        self.reasoner = reasoner
        self.terminology = terminology
        self.agent_settings = agent_settings
        self.generation_settings = generation_settings
        self.citation_verifier = citation_verifier or CitationVerifier()
        self.context_builder = EvidenceContextBuilder(
            max_evidence=generation_settings.max_context_evidence,
            max_per_paper=generation_settings.max_evidence_per_paper,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(ResearchState)
        builder.add_node("classify_question", self._classify_question)
        builder.add_node("expand_domain_terms", self._expand_domain_terms)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("grade_evidence", self._grade_evidence)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("verify_citations", self._verify_citations)
        builder.add_node("repair_answer", self._repair_answer)
        builder.add_node("complete", self._complete)
        builder.add_node("abstain", self._abstain)

        builder.add_edge(START, "classify_question")
        builder.add_edge("classify_question", "expand_domain_terms")
        builder.add_edge("expand_domain_terms", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            self._route_after_grade,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "abstain": "abstain",
            },
        )
        builder.add_conditional_edges(
            "rewrite_query",
            self._route_after_rewrite,
            {"retrieve_evidence": "retrieve_evidence", "abstain": "abstain"},
        )
        builder.add_conditional_edges(
            "generate_answer",
            self._route_after_generation,
            {"verify_citations": "verify_citations", "abstain": "abstain"},
        )
        builder.add_conditional_edges(
            "verify_citations",
            self._route_after_verification,
            {
                "complete": "complete",
                "repair_answer": "repair_answer",
                "abstain": "abstain",
            },
        )
        builder.add_edge("repair_answer", "verify_citations")
        builder.add_edge("complete", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def run(self, question: str, *, paper_ids: list[str] | None = None) -> AgentRun:
        """Execute the graph and return its answer plus auditable transitions."""
        final_run: AgentRun | None = None
        for item in self.stream(question, paper_ids=paper_ids):
            if isinstance(item, AgentRun):
                final_run = item
        if final_run is None:
            raise RuntimeError("Agent graph completed without a final result")
        return final_run

    def stream(
        self, question: str, *, paper_ids: list[str] | None = None
    ) -> Iterator[TraceEvent | AgentRun]:
        """Yield safe trace events as nodes complete, followed by the final run."""
        initial, recursion_limit = self._initial_state(question, paper_ids)
        final = initial
        emitted_events = 0
        for snapshot in self.graph.stream(
            initial,
            {"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            final = cast(ResearchState, snapshot)
            events = final.get("trace_events", [])
            yield from events[emitted_events:]
            emitted_events = len(events)
        yield self._run_from_state(final)

    def _initial_state(
        self, question: str, paper_ids: list[str] | None
    ) -> tuple[ResearchState, int]:
        """Validate input and create a fresh state and recursion bound."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        initial: ResearchState = {
            "question": question.strip(),
            "selected_paper_ids": list(paper_ids or []),
            "search_queries": [],
            "rewrite_reasons": [],
            "round_queries": [],
            "retrieved_evidence": [],
            "selected_evidence": [],
            "missing_aspects": [],
            "retry_count": 0,
            "repair_count": 0,
            "rewrite_valid": True,
            "citation_valid": False,
            "verification_errors": [],
            "trace_events": [],
        }
        recursion_limit = (
            12
            + self.agent_settings.max_retrieval_retries * 3
            + self.agent_settings.max_answer_repairs * 2
        )
        return initial, recursion_limit

    @staticmethod
    def _run_from_state(final: ResearchState) -> AgentRun:
        """Convert the terminal graph state to the public run contract."""
        return AgentRun(
            answer=final["result"],
            question_type=final["plan"].question_type,
            plan=final["plan"],
            search_queries=final["search_queries"],
            rewrite_reasons=final["rewrite_reasons"],
            retrieval_retries=final["retry_count"],
            answer_repairs=final["repair_count"],
            trace_events=final["trace_events"],
        )

    def _classify_question(self, state: ResearchState) -> ResearchState:
        plan = self.reasoner.classify_and_plan(state["question"])
        events = self._append_trace(
            state,
            event="question_classified",
            node="classify_question",
            summary=f"Question classified as {plan.question_type.value}.",
        )
        events = self._append_event(
            events,
            event="retrieval_planned",
            node="classify_question",
            summary=(
                f"Planned {len(plan.sub_questions)} sub-questions and "
                f"{len(plan.preferred_sections)} preferred sections."
            ),
        )
        return {
            "plan": plan,
            "trace_events": events,
        }

    def _expand_domain_terms(self, state: ResearchState) -> ResearchState:
        bases = [state["question"], *state["plan"].sub_questions]
        queries: list[str] = []
        seen: set[str] = set()
        for base in bases:
            for query in expand_query(base, self.terminology):
                normalized = query.casefold().strip()
                if normalized in seen:
                    continue
                seen.add(normalized)
                queries.append(query)
                if len(queries) >= self.agent_settings.max_queries_per_round:
                    break
            if len(queries) >= self.agent_settings.max_queries_per_round:
                break
        return {
            "search_queries": queries,
            "round_queries": queries,
            "trace_events": self._append_trace(
                state,
                event="query_expanded",
                node="expand_domain_terms",
                summary=f"Prepared {len(queries)} bounded retrieval queries.",
            ),
        }

    def _retrieve_evidence(self, state: ResearchState) -> ResearchState:
        events = self._append_trace(
            state,
            event="retrieval_started",
            node="retrieve_evidence",
            summary=f"Started {len(state['round_queries'])} local retrieval queries.",
        )
        combined = list(state.get("retrieved_evidence", []))
        seen = {item.id for item in combined}
        for query in state["round_queries"]:
            results = self.retriever.search(
                query,
                paper_ids=state["selected_paper_ids"] or None,
            )
            for item in results:
                if item.id not in seen:
                    combined.append(item)
                    seen.add(item.id)

        preferred = {value.casefold() for value in state["plan"].preferred_sections}
        if preferred:
            combined.sort(
                key=lambda item: 0
                if item.section and item.section.casefold() in preferred
                else 1
            )
        return {
            "retrieved_evidence": combined,
            "trace_events": self._append_event(
                events,
                event="evidence_retrieved",
                node="retrieve_evidence",
                summary=(
                    f"Retrieved {len(combined)} unique evidence items after "
                    f"{state.get('retry_count', 0)} retries."
                ),
            ),
        }

    def _grade_evidence(self, state: ResearchState) -> ResearchState:
        evidence = self.context_builder.select(state["retrieved_evidence"])
        if not evidence:
            grade = EvidenceGrade(
                sufficient=False,
                missing_aspects=["No relevant local-paper evidence was retrieved."],
            )
        else:
            grade = self.reasoner.grade_evidence(
                state["question"],
                evidence,
                selected_paper_ids=state["selected_paper_ids"],
                requires_comparison=state["plan"].requires_comparison,
            )

        evidence_by_id = {item.id: item for item in evidence}
        selected = []
        seen_selected: set[str] = set()
        invalid_selection = False
        for evidence_id in grade.selected_evidence_ids:
            if evidence_id not in evidence_by_id or evidence_id in seen_selected:
                invalid_selection = True
                continue
            selected.append(evidence_by_id[evidence_id])
            seen_selected.add(evidence_id)
        if grade.sufficient and invalid_selection:
            grade = EvidenceGrade(
                sufficient=False,
                selected_evidence_ids=[item.id for item in selected],
                missing_aspects=[
                    "The evidence grader selected unavailable evidence IDs."
                ],
            )
        return {
            "grade": grade,
            "selected_evidence": selected,
            "missing_aspects": grade.missing_aspects,
            "trace_events": self._append_trace(
                state,
                event="evidence_graded",
                node="grade_evidence",
                summary=(
                    f"Evidence sufficient: {grade.sufficient}; "
                    f"selected {len(selected)} items."
                ),
            ),
        }

    def _rewrite_query(self, state: ResearchState) -> ResearchState:
        rewrite = self.reasoner.rewrite_query(
            state["question"],
            previous_queries=state["search_queries"],
            missing_aspects=state["missing_aspects"],
        )
        normalized_previous = {
            query.casefold().strip() for query in state["search_queries"]
        }
        valid = rewrite.query.casefold().strip() not in normalized_previous
        queries = list(state["search_queries"])
        rewrite_reasons = list(state["rewrite_reasons"])
        retry_count = state["retry_count"]
        if valid:
            queries.append(rewrite.query.strip())
            rewrite_reasons.append(rewrite.reason.strip())
            retry_count += 1
        return {
            "search_queries": queries,
            "rewrite_reasons": rewrite_reasons,
            "round_queries": [rewrite.query.strip()] if valid else [],
            "retry_count": retry_count,
            "rewrite_valid": valid,
            "verification_errors": []
            if valid
            else ["Query rewrite repeated a previous query."],
            "trace_events": self._append_trace(
                state,
                event="query_rewritten",
                node="rewrite_query",
                summary=(
                    f"Query rewrite accepted for retry {retry_count}."
                    if valid
                    else "Rejected a duplicate query rewrite."
                ),
            ),
        }

    def _generate_answer(self, state: ResearchState) -> ResearchState:
        draft = self.generator.generate(state["question"], state["selected_evidence"])
        return {
            "answer_draft": draft,
            "missing_aspects": draft.missing_evidence,
            "trace_events": self._append_trace(
                state,
                event="answer_generating",
                node="generate_answer",
                summary=(
                    f"Generated {len(draft.claims)} grounded claim candidates."
                    if draft.answerable
                    else "Generation reported an evidence gap."
                ),
            ),
        }

    def _verify_citations(self, state: ResearchState) -> ResearchState:
        draft = state["answer_draft"]
        if not draft.answerable:
            valid = False
            errors = ["Answer repair left no supported claims."]
        else:
            integrity = self.citation_verifier.verify(draft, state["selected_evidence"])
            if not integrity.valid:
                valid = False
                errors = integrity.errors
            else:
                semantic = self.reasoner.verify_citations(
                    state["question"], draft, state["selected_evidence"]
                )
                valid = semantic.supported
                errors = semantic.errors
        return {
            "citation_valid": valid,
            "verification_errors": errors,
            "trace_events": self._append_trace(
                state,
                event="citation_verifying",
                node="verify_citations",
                summary=(
                    "Citation integrity and semantic support passed."
                    if valid
                    else f"Citation verification found {len(errors)} issue(s)."
                ),
            ),
        }

    def _repair_answer(self, state: ResearchState) -> ResearchState:
        repaired = self.reasoner.repair_answer(
            state["question"],
            state["answer_draft"],
            state["selected_evidence"],
            verification_errors=state["verification_errors"],
        )
        repair_count = state["repair_count"] + 1
        return {
            "answer_draft": repaired,
            "repair_count": repair_count,
            "missing_aspects": repaired.missing_evidence,
            "trace_events": self._append_trace(
                state,
                event="answer_generating",
                node="repair_answer",
                summary=f"Applied bounded answer repair {repair_count}.",
            ),
        }

    def _complete(self, state: ResearchState) -> ResearchState:
        verification = self.citation_verifier.verify(
            state["answer_draft"], state["selected_evidence"]
        )
        result = GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            answer=render_claims(state["answer_draft"]),
            citations=verification.citations,
            evidence=state["selected_evidence"],
            searched_paper_ids=state["selected_paper_ids"],
        )
        return {
            "result": result,
            "trace_events": self._append_trace(
                state,
                event="completed",
                node="complete",
                summary=f"Completed with {len(result.citations)} verified citations.",
            ),
        }

    def _abstain(self, state: ResearchState) -> ResearchState:
        missing = state.get("missing_aspects", [])
        errors = state.get("verification_errors", [])
        if errors and not missing:
            missing = ["The available evidence could not support a verified answer."]
        result = GroundedAnswer(
            status=AnswerStatus.ABSTAINED,
            answer=self.generation_settings.refusal_message,
            evidence=self.context_builder.select(
                state.get("selected_evidence") or state.get("retrieved_evidence", [])
            ),
            missing_evidence=missing,
            verification_errors=errors,
            searched_paper_ids=state["selected_paper_ids"],
        )
        return {
            "result": result,
            "trace_events": self._append_trace(
                state,
                event="completed",
                node="abstain",
                summary="Completed with an explicit evidence-insufficient refusal.",
            ),
        }

    def _route_after_grade(
        self, state: ResearchState
    ) -> Literal["generate_answer", "rewrite_query", "abstain"]:
        if state["grade"].sufficient:
            return "generate_answer"
        if state["retry_count"] < self.agent_settings.max_retrieval_retries:
            return "rewrite_query"
        if state.get("selected_evidence"):
            # A conservative grader may still have selected useful evidence. Let the
            # generator and both citation checks make the final fail-closed decision.
            return "generate_answer"
        return "abstain"

    @staticmethod
    def _route_after_rewrite(
        state: ResearchState,
    ) -> Literal["retrieve_evidence", "abstain"]:
        return "retrieve_evidence" if state["rewrite_valid"] else "abstain"

    @staticmethod
    def _route_after_generation(
        state: ResearchState,
    ) -> Literal["verify_citations", "abstain"]:
        return "verify_citations" if state["answer_draft"].answerable else "abstain"

    def _route_after_verification(
        self, state: ResearchState
    ) -> Literal["complete", "repair_answer", "abstain"]:
        if state["citation_valid"]:
            return "complete"
        if state["repair_count"] < self.agent_settings.max_answer_repairs:
            return "repair_answer"
        return "abstain"

    @staticmethod
    def _append_trace(
        state: ResearchState, *, event: str, node: str, summary: str
    ) -> list[TraceEvent]:
        events = list(state.get("trace_events", []))
        return AgenticRAGWorkflow._append_event(
            events, event=event, node=node, summary=summary
        )

    @staticmethod
    def _append_event(
        events: list[TraceEvent], *, event: str, node: str, summary: str
    ) -> list[TraceEvent]:
        events = list(events)
        events.append(
            TraceEvent(
                sequence=len(events) + 1,
                event=event,
                node=node,
                summary=summary,
            )
        )
        return events
