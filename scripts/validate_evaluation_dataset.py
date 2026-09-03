"""Validate an evaluation JSONL file against the current local corpus."""

import argparse
import json
import sys
from pathlib import Path

from hardsec_scholar.api.services import LocalApplicationServices
from hardsec_scholar.evaluation import (
    load_evaluation_dataset,
    validate_evaluation_dataset,
)


def main() -> None:
    """Load the configured benchmark and print its validation report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/evaluations/hardsec_benchmark_v1.jsonl"),
    )
    args = parser.parse_args()
    services = LocalApplicationServices()
    samples = load_evaluation_dataset(args.dataset)
    report = validate_evaluation_dataset(samples, services.repository)
    sys.stdout.write(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
