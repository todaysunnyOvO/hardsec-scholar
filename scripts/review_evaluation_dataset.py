"""Semantically review benchmark reference answers against their gold chunks."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_openai import ChatOpenAI

from hardsec_scholar.agent import SemanticCitationCheck, StructuredAgentReasoner
from hardsec_scholar.api.services import LocalApplicationServices
from hardsec_scholar.domain import Evidence
from hardsec_scholar.evaluation import EvaluationSample, load_evaluation_dataset
from hardsec_scholar.generation import AnswerDraft, ClaimDraft


def review_sample(
    sample: EvaluationSample,
    services: LocalApplicationServices,
    reasoner: StructuredAgentReasoner,
) -> tuple[str, SemanticCitationCheck]:
    """Check one reference answer using only its declared gold chunks."""
    evidence: list[Evidence] = []
    for chunk_id in sample.relevant_chunk_ids:
        chunk = services.repository.get_chunk(chunk_id)
        if chunk is None:
            raise ValueError(f"Unknown chunk {chunk_id} in {sample.id}")
        evidence.append(
            Evidence(
                id=chunk.id,
                chunk_id=chunk.id,
                paper_id=chunk.paper_id,
                paper_title=chunk.title,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
            )
        )
    draft = AnswerDraft(
        answerable=True,
        claims=[
            ClaimDraft(
                text=sample.reference_answer,
                evidence_ids=sample.relevant_chunk_ids,
            )
        ],
    )
    return sample.id, reasoner.verify_citations(sample.question, draft, evidence)


def main() -> None:
    """Review every answerable sample and return a machine-readable report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/evaluations/hardsec_benchmark_v1.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    services = LocalApplicationServices()
    runtime = services.runtime
    if runtime.llm_api_key is None:
        raise ValueError("LLM_API_KEY is required for semantic review")
    extra_body = None
    if runtime.llm_base_url and "deepseek.com" in runtime.llm_base_url:
        extra_body = {"thinking": {"type": "disabled"}}
    model = ChatOpenAI(
        model=runtime.llm_model,
        api_key=runtime.llm_api_key,
        base_url=runtime.llm_base_url,
        temperature=0,
        extra_body=extra_body,
    )
    reasoner = StructuredAgentReasoner(model)
    samples = [
        sample
        for sample in load_evaluation_dataset(args.dataset)
        if not sample.should_abstain
    ]

    results: dict[str, SemanticCitationCheck] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(review_sample, sample, services, reasoner)
            for sample in samples
        ]
        for future in as_completed(futures):
            sample_id, check = future.result()
            results[sample_id] = check

    report = {
        "reviewed": len(results),
        "supported": sum(check.supported for check in results.values()),
        "results": {
            sample_id: check.model_dump(mode="json")
            for sample_id, check in sorted(results.items())
        },
    }
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    if report["supported"] != report["reviewed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
