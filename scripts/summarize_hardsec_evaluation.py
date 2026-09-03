"""Summarize persisted HardSec Scholar evaluation runs."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from hardsec_scholar.evaluation import EvaluationRunResult, EvaluationVariant


def mean(values: list[float]) -> float | None:
    """Return a rounded mean while preserving missing metric groups."""
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile for a non-empty latency list."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.9999) - 1))
    return round(ordered[index], 4)


def metric_values(
    runs: list[EvaluationRunResult], section: str, field: str
) -> list[float]:
    """Collect present numeric metrics from successful result records."""
    values: list[float] = []
    for run in runs:
        metrics = getattr(run, section)
        if metrics is None:
            continue
        value = getattr(metrics, field)
        if value is not None:
            values.append(float(value))
    return values


def summarize(runs: list[EvaluationRunResult]) -> dict[str, Any]:
    """Aggregate retrieval, answer, latency, token, and loop metrics by variant."""
    attempts: dict[EvaluationVariant, list[EvaluationRunResult]] = defaultdict(list)
    latest: dict[tuple[str, EvaluationVariant], EvaluationRunResult] = {}
    for run in runs:
        attempts[run.variant].append(run)
        latest[(run.sample_id, run.variant)] = run
    grouped: dict[EvaluationVariant, list[EvaluationRunResult]] = defaultdict(list)
    for run in latest.values():
        grouped[run.variant].append(run)
    summary: dict[str, Any] = {}
    for variant, group in grouped.items():
        successful = [run for run in group if run.success]
        latencies = [run.latency_seconds for run in successful]
        variant_attempts = attempts[variant]
        summary[variant.value] = {
            "attempts": len(variant_attempts),
            "runs": len(group),
            "successful": len(successful),
            "failed": len(group) - len(successful),
            "retrieval": {
                field: mean(metric_values(successful, "retrieval", field))
                for field in [
                    "recall_at_5",
                    "recall_at_10",
                    "mrr",
                    "paper_hit",
                    "page_hit",
                    "all_round_recall",
                ]
            },
            "answer": {
                field: mean(metric_values(successful, "answer_metrics", field))
                for field in [
                    "abstention_correct",
                    "reference_token_f1",
                    "citation_precision",
                    "citation_recall",
                    "citation_page_precision",
                    "citation_paper_precision",
                    "grounded",
                ]
            },
            "latency_seconds": {
                "mean": mean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
            },
            "usage": {
                "input_tokens": sum(run.usage.input_tokens for run in successful),
                "output_tokens": sum(run.usage.output_tokens for run in successful),
                "total_tokens": sum(run.usage.total_tokens for run in successful),
                "estimated_usd": round(
                    sum(run.usage.estimated_usd for run in successful), 6
                ),
                "mean_estimated_usd": (
                    round(
                        sum(run.usage.estimated_usd for run in successful)
                        / len(successful),
                        6,
                    )
                    if successful
                    else None
                ),
            },
            "attempt_usage": {
                "input_tokens": sum(
                    run.usage.input_tokens for run in variant_attempts
                ),
                "output_tokens": sum(
                    run.usage.output_tokens for run in variant_attempts
                ),
                "total_tokens": sum(
                    run.usage.total_tokens for run in variant_attempts
                ),
                "estimated_usd": round(
                    sum(run.usage.estimated_usd for run in variant_attempts), 6
                ),
            },
            "agent": {
                "mean_queries": mean(
                    [float(run.query_count) for run in successful]
                ),
                "retrieval_retry_rate": mean(
                    [float(run.retrieval_retries > 0) for run in successful]
                ),
                "mean_retrieval_retries": mean(
                    [float(run.retrieval_retries) for run in successful]
                ),
                "answer_repair_rate": mean(
                    [float(run.answer_repairs > 0) for run in successful]
                ),
            },
        }
    return summary


def main() -> None:
    """Read JSONL results and print or persist the aggregate summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path("data/evaluations/results/hardsec_v1_runs.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.results.open(encoding="utf-8") as stream:
        runs = [
            EvaluationRunResult.model_validate_json(line)
            for line in stream
            if line.strip()
        ]
    rendered = json.dumps(summarize(runs), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    sys.stdout.write(rendered)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
