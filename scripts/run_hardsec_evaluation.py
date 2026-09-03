"""Run resumable Baseline and Agentic RAG experiments on the local benchmark."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Protocol

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_openai import ChatOpenAI

from hardsec_scholar.agent import AgenticRAGWorkflow, StructuredAgentReasoner
from hardsec_scholar.api.services import LocalApplicationServices
from hardsec_scholar.domain import Evidence
from hardsec_scholar.domain.terminology import load_terminology
from hardsec_scholar.evaluation import (
    EvaluationRunResult,
    EvaluationSample,
    EvaluationVariant,
    TokenUsage,
    evaluate_answer,
    evaluate_retrieval,
    load_evaluation_dataset,
    unique_evidence,
    validate_evaluation_dataset,
)
from hardsec_scholar.generation import (
    BasicRAGService,
    GroundedAnswer,
    StructuredAnswerGenerator,
)
from hardsec_scholar.retrieval import DenseRetriever, HybridRetriever
from hardsec_scholar.retrieval.reranker import FlashRankReranker


class Retriever(Protocol):
    """Describe the retrieval method wrapped by the run recorder."""

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        """Return ranked evidence."""
        ...


class RecordingRetriever:
    """Record every search result without changing retrieval behavior."""

    def __init__(self, delegate: Retriever) -> None:
        """Wrap one dense or hybrid retriever."""
        self.delegate = delegate
        self.calls: list[list[Evidence]] = []

    def reset(self) -> None:
        """Clear evidence from the previous evaluation sample."""
        self.calls.clear()

    def search(
        self, query: str, *, paper_ids: list[str] | None = None
    ) -> list[Evidence]:
        """Delegate retrieval and retain its ranked result."""
        result = self.delegate.search(query, paper_ids=paper_ids)
        self.calls.append(result)
        return result

    @property
    def first_search(self) -> list[Evidence]:
        """Return the initial query ranking used for Recall@k and MRR."""
        return self.calls[0] if self.calls else []

    @property
    def all_searches(self) -> list[Evidence]:
        """Return the first-seen union across every query and retry."""
        return unique_evidence([item for call in self.calls for item in call])


def parse_args() -> argparse.Namespace:
    """Parse dataset, resume, variant, and explicit pricing controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/hardsec_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/results/hardsec_v1_runs.jsonl"),
    )
    parser.add_argument(
        "--variant",
        choices=["both", "baseline", "agentic"],
        default="both",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--input-usd-per-million", type=float, default=0.14)
    parser.add_argument("--output-usd-per-million", type=float, default=0.28)
    return parser.parse_args()


def create_chat_model(services: LocalApplicationServices) -> ChatOpenAI:
    """Create the application's configured structured-output chat model."""
    runtime = services.runtime
    if runtime.llm_api_key is None:
        raise ValueError("LLM_API_KEY is required to run the evaluation")
    extra_body = None
    if runtime.llm_base_url and "deepseek.com" in runtime.llm_base_url:
        extra_body = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        model=runtime.llm_model,
        api_key=runtime.llm_api_key,
        base_url=runtime.llm_base_url,
        temperature=0,
        extra_body=extra_body,
    )


def token_usage(
    metadata: dict[str, dict[str, int]],
    *,
    input_rate: float,
    output_rate: float,
) -> TokenUsage:
    """Aggregate provider usage and apply explicit cache-miss price rates."""
    input_tokens = sum(int(item.get("input_tokens", 0)) for item in metadata.values())
    output_tokens = sum(
        int(item.get("output_tokens", 0)) for item in metadata.values()
    )
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in metadata.values())
    estimated = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
        estimated_usd=estimated,
    )


def cited_chunks(answer: GroundedAnswer) -> list[str]:
    """Resolve citation evidence IDs back to stable corpus chunk IDs."""
    mapping = {item.id: item.chunk_id for item in answer.evidence}
    return list(
        dict.fromkeys(
            mapping[citation.evidence_id]
            for citation in answer.citations
            if citation.evidence_id in mapping
        )
    )


def completed_keys(path: Path) -> set[tuple[str, EvaluationVariant]]:
    """Read successful persisted keys while leaving failed attempts retryable."""
    if not path.is_file():
        return set()
    keys: set[tuple[str, EvaluationVariant]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            result = EvaluationRunResult.model_validate_json(line)
            if result.success:
                keys.add((result.sample_id, result.variant))
    return keys


def run_variant(
    sample: EvaluationSample,
    variant: EvaluationVariant,
    recorder: RecordingRetriever,
    service: BasicRAGService | AgenticRAGWorkflow,
    *,
    input_rate: float,
    output_rate: float,
) -> EvaluationRunResult:
    """Execute one system, collect usage, and score deterministic metrics."""
    recorder.reset()
    started = time.perf_counter()
    error: Exception | None = None
    answer: GroundedAnswer | None = None
    retries = 0
    repairs = 0
    query_count = 0
    with get_usage_metadata_callback() as callback:
        try:
            if variant is EvaluationVariant.BASELINE:
                answer = service.answer(sample.question)  # type: ignore[union-attr]
                query_count = 1
            else:
                agent_run = service.run(sample.question)  # type: ignore[union-attr]
                answer = agent_run.answer
                retries = agent_run.retrieval_retries
                repairs = agent_run.answer_repairs
                query_count = len(agent_run.search_queries)
        except Exception as exc:
            error = exc
    usage = token_usage(
        callback.usage_metadata,  # type: ignore[arg-type]
        input_rate=input_rate,
        output_rate=output_rate,
    )
    if error is not None or answer is None:
        return EvaluationRunResult(
            sample_id=sample.id,
            variant=variant,
            success=False,
            error=(
                f"{type(error).__name__}: {str(error)[:800]}"
                if error is not None
                else "RuntimeError: Evaluation completed without an answer"
            ),
            latency_seconds=time.perf_counter() - started,
            usage=usage,
        )
    all_evidence = recorder.all_searches
    return EvaluationRunResult(
        sample_id=sample.id,
        variant=variant,
        success=True,
        latency_seconds=time.perf_counter() - started,
        answer_status=answer.status.value,
        answer=answer.answer,
        retrieved_chunk_ids=[item.chunk_id for item in all_evidence],
        cited_chunk_ids=cited_chunks(answer),
        retrieval=evaluate_retrieval(sample, recorder.first_search, all_evidence),
        answer_metrics=evaluate_answer(sample, answer),
        usage=usage,
        retrieval_retries=retries,
        answer_repairs=repairs,
        query_count=query_count,
    )


def main() -> None:
    """Run requested experiment pairs and append each result immediately."""
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.input_usd_per_million < 0 or args.output_usd_per_million < 0:
        raise ValueError("Token prices cannot be negative")
    if args.force and args.output.exists():
        args.output.unlink()

    services = LocalApplicationServices()
    samples = load_evaluation_dataset(args.dataset)
    validate_evaluation_dataset(samples, services.repository)
    if args.sample_id:
        requested_ids = set(args.sample_id)
        known_ids = {sample.id for sample in samples}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"Unknown --sample-id values: {unknown_ids}")
        samples = [sample for sample in samples if sample.id in requested_ids]
    if args.limit is not None:
        samples = samples[: args.limit]

    model = create_chat_model(services)
    vector_index = services._vector_index()
    baseline_recorder = RecordingRetriever(DenseRetriever(vector_index, top_k=10))
    baseline = BasicRAGService(
        retriever=baseline_recorder,
        generator=StructuredAnswerGenerator(model),
        settings=services.settings.generation,
    )

    eval_retrieval = services.settings.retrieval.model_copy(
        update={"rerank_top_k": 10}
    )
    hybrid = HybridRetriever(
        repository=services.repository,
        vector_index=vector_index,
        reranker=FlashRankReranker(
            model_name=eval_retrieval.reranker_model,
            cache_dir=eval_retrieval.reranker_cache,
        ),
        settings=eval_retrieval,
    )
    agent_recorder = RecordingRetriever(hybrid)
    agent = AgenticRAGWorkflow(
        retriever=agent_recorder,
        generator=StructuredAnswerGenerator(model),
        reasoner=StructuredAgentReasoner(model),
        terminology=load_terminology(),
        agent_settings=services.settings.agent,
        generation_settings=services.settings.generation,
    )

    variants = {
        "both": [EvaluationVariant.BASELINE, EvaluationVariant.AGENTIC],
        "baseline": [EvaluationVariant.BASELINE],
        "agentic": [EvaluationVariant.AGENTIC],
    }[args.variant]
    recorders = {
        EvaluationVariant.BASELINE: baseline_recorder,
        EvaluationVariant.AGENTIC: agent_recorder,
    }
    systems: dict[EvaluationVariant, BasicRAGService | AgenticRAGWorkflow] = {
        EvaluationVariant.BASELINE: baseline,
        EvaluationVariant.AGENTIC: agent,
    }
    done = set() if args.rerun_successful else completed_keys(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(samples) * len(variants)
    completed = 0
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            for variant in variants:
                key = (sample.id, variant)
                if key in done:
                    completed += 1
                    continue
                result = run_variant(
                    sample,
                    variant,
                    recorders[variant],
                    systems[variant],
                    input_rate=args.input_usd_per_million,
                    output_rate=args.output_usd_per_million,
                )
                stream.write(
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
                )
                stream.write("\n")
                stream.flush()
                completed += 1
                sys.stdout.write(
                    f"[{completed}/{total}] {sample.id} {variant.value} "
                    f"success={result.success} latency={result.latency_seconds:.2f}s\n"
                )
                sys.stdout.flush()


if __name__ == "__main__":
    main()
