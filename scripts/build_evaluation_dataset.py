"""Build the reviewed HardSec Scholar benchmark from the indexed paper corpus."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from hardsec_scholar.api.services import LocalApplicationServices
from hardsec_scholar.domain import PaperMetadata
from hardsec_scholar.evaluation import (
    EvaluationCategory,
    EvaluationSample,
    validate_evaluation_dataset,
)


class GeneratedQuestion(BaseModel):
    """Hold one model-proposed question before corpus validation."""

    model_config = ConfigDict(extra="forbid")

    category: EvaluationCategory
    question: str = Field(min_length=5)
    reference_answer: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1, max_length=4)


class GeneratedQuestionBatch(BaseModel):
    """Collect all requested questions from one bounded evidence context."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GeneratedQuestion]


PAPER_ASSIGNMENTS = {
    "Hardware-Software Contracts": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.THREAT_MODEL,
    ],
    "Revizor": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.MECHANISM,
        EvaluationCategory.EXPERIMENT,
    ],
    "Hide and Seek with Spectres": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.MECHANISM,
        EvaluationCategory.EXPERIMENT,
    ],
    "Speculation at Fault": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.MECHANISM,
    ],
    "AMuLeT": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.THREAT_MODEL,
        EvaluationCategory.EXPERIMENT,
    ],
    "Enter, Exit, Page Fault": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.THREAT_MODEL,
    ],
    "Adversarial Prefetch": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.MECHANISM,
    ],
    "AVX Timing": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.THREAT_MODEL,
    ],
    "What the Fuzz": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.EXPERIMENT,
    ],
    "MEMORY DISORDER": [
        EvaluationCategory.SINGLE_PAPER_FACT,
        EvaluationCategory.MECHANISM,
    ],
}

COMPARISON_PAIRS = [
    ("Revizor", "Hide and Seek with Spectres"),
    ("Hardware-Software Contracts", "AMuLeT"),
    ("Adversarial Prefetch", "AVX Timing"),
    ("Speculation at Fault", "Enter, Exit, Page Fault"),
]

UNANSWERABLE_QUESTIONS = [
    (
        "Which FPGA board and synthesis tool do the indexed papers use to implement "
        "a post-quantum cryptography accelerator?"
    ),
    (
        "What detection accuracy do the indexed papers report for finding hardware "
        "Trojans from thermal-camera images?"
    ),
    (
        "Which fabrication node and die area do the indexed papers report for a "
        "RISC-V cryptographic accelerator?"
    ),
]

SYSTEM_PROMPT = """Create evaluation questions for a hardware-security paper RAG system.
Use only the supplied evidence blocks. Produce exactly one question for every requested
category. Questions and reference answers must be in English. Each reference answer must
be fully and directly supported by one to four exact chunk IDs from the context. Prefer
specific, technically meaningful questions over title-level trivia. Do not cite a chunk
unless it supports the whole reference answer. Do not invent measurements, platforms,
attacker capabilities, or conclusions."""


def parse_args() -> argparse.Namespace:
    """Parse output and overwrite controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/hardsec_benchmark_v1.jsonl"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_paper(
    services: LocalApplicationServices, title_fragment: str
) -> PaperMetadata:
    """Resolve one configured title fragment to exactly one indexed paper."""
    matches = [
        paper
        for paper in services.repository.list_papers()
        if title_fragment.casefold() in paper.title.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one paper matching {title_fragment!r}, found {len(matches)}"
        )
    return matches[0]


def render_chunks(services: LocalApplicationServices, paper_ids: list[str]) -> str:
    """Render stable chunk IDs and page metadata for bounded generation."""
    blocks = []
    for chunk in services.repository.list_chunks(paper_ids):
        blocks.append(
            f'<chunk id="{chunk.id}" paper_id="{chunk.paper_id}" '
            f'pages="{chunk.page_start}-{chunk.page_end}" '
            f'section="{chunk.section or "Unknown"}">\n{chunk.text}\n</chunk>'
        )
    return "\n\n".join(blocks)


def create_model(services: LocalApplicationServices) -> Any:
    """Create the same portable non-thinking structured model used by the app."""
    runtime = services.runtime
    if runtime.llm_api_key is None:
        raise ValueError("LLM_API_KEY is required to build the evaluation dataset")
    extra_body = None
    if runtime.llm_base_url and "deepseek.com" in runtime.llm_base_url:
        extra_body = {"thinking": {"type": "disabled"}}
    chat = ChatOpenAI(
        model=runtime.llm_model,
        api_key=runtime.llm_api_key,
        base_url=runtime.llm_base_url,
        temperature=0,
        extra_body=extra_body,
    )
    return chat.with_structured_output(
        GeneratedQuestionBatch, method="function_calling"
    )


def generate_batch(
    model: Any,
    services: LocalApplicationServices,
    paper_ids: list[str],
    categories: list[EvaluationCategory],
) -> list[GeneratedQuestion]:
    """Generate and structurally check one bounded group of gold candidates."""
    context = render_chunks(services, paper_ids)
    requested = [category.value for category in categories]
    chunks = services.repository.list_chunks(paper_ids)
    chunk_papers = {chunk.id: chunk.paper_id for chunk in chunks}
    comparison_instruction = ""
    if EvaluationCategory.COMPARISON in categories:
        comparison_instruction = (
            "A comparison question must contrast every allowed paper and cite at "
            "least one supporting chunk from each paper.\n"
        )

    last_categories: list[EvaluationCategory] = []
    for attempt in range(1, 4):
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Requested categories: {requested}\n"
                        f"Required question count: {len(categories)}\n"
                        f"Allowed paper IDs: {paper_ids}\n"
                        f"{comparison_instruction}"
                        f"Generation attempt: {attempt}/3\n\nEvidence:\n{context}"
                    )
                ),
            ]
        )
        batch = GeneratedQuestionBatch.model_validate(response)
        selected: dict[EvaluationCategory, GeneratedQuestion] = {}
        for question in batch.questions:
            if question.category not in categories or question.category in selected:
                continue
            if any(chunk_id not in chunk_papers for chunk_id in question.relevant_chunk_ids):
                continue
            cited_papers = {
                chunk_papers[chunk_id] for chunk_id in question.relevant_chunk_ids
            }
            if (
                question.category is EvaluationCategory.COMPARISON
                and cited_papers != set(paper_ids)
            ):
                continue
            selected[question.category] = question
        if set(selected) == set(categories):
            return [selected[category] for category in categories]
        last_categories = list(selected)

    raise ValueError(
        f"Model produced valid categories {last_categories}, expected {categories}"
    )


def materialize_sample(
    sequence: int,
    candidate: GeneratedQuestion,
    paper_ids: list[str],
    services: LocalApplicationServices,
) -> EvaluationSample:
    """Derive exact expected pages from the candidate's stable chunk IDs."""
    pages: set[int] = set()
    chunk_papers: set[str] = set()
    for chunk_id in candidate.relevant_chunk_ids:
        chunk = services.repository.get_chunk(chunk_id)
        if chunk is None:
            raise ValueError(f"Unknown generated chunk ID: {chunk_id}")
        chunk_papers.add(chunk.paper_id)
        pages.update(range(chunk.page_start, chunk.page_end + 1))
    if candidate.category is EvaluationCategory.COMPARISON:
        if chunk_papers != set(paper_ids):
            raise ValueError("Comparison evidence must cover both configured papers")
    return EvaluationSample(
        id=f"hardsec_eval_{sequence:03d}",
        question=candidate.question,
        category=candidate.category,
        paper_ids=paper_ids,
        reference_answer=candidate.reference_answer,
        relevant_chunk_ids=list(dict.fromkeys(candidate.relevant_chunk_ids)),
        expected_pages=sorted(pages),
    )


def build_dataset(
    services: LocalApplicationServices, model: Any
) -> list[EvaluationSample]:
    """Generate the approved 30-question benchmark shape."""
    samples: list[EvaluationSample] = []
    sequence = 1
    for title_fragment, categories in PAPER_ASSIGNMENTS.items():
        paper = find_paper(services, title_fragment)
        candidates = generate_batch(model, services, [paper.id], categories)
        for candidate in candidates:
            samples.append(
                materialize_sample(sequence, candidate, [paper.id], services)
            )
            sequence += 1

    for left_title, right_title in COMPARISON_PAIRS:
        paper_ids = [
            find_paper(services, left_title).id,
            find_paper(services, right_title).id,
        ]
        candidate = generate_batch(
            model, services, paper_ids, [EvaluationCategory.COMPARISON]
        )[0]
        samples.append(materialize_sample(sequence, candidate, paper_ids, services))
        sequence += 1

    for question in UNANSWERABLE_QUESTIONS:
        samples.append(
            EvaluationSample(
                id=f"hardsec_eval_{sequence:03d}",
                question=question,
                category=EvaluationCategory.UNANSWERABLE,
                reference_answer=(
                    "The indexed hardware-security papers do not provide this information."
                ),
                should_abstain=True,
            )
        )
        sequence += 1
    return samples


def main() -> None:
    """Build, validate, and write the evaluation JSONL deterministically."""
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")
    services = LocalApplicationServices()
    samples = build_dataset(services, create_model(services))
    report = validate_evaluation_dataset(samples, services.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False))
            stream.write("\n")
    sys.stdout.write(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
