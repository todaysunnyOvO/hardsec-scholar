"""Offline evaluation dataset contracts and validation."""

from hardsec_scholar.evaluation.dataset import (
    DatasetValidationReport,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)
from hardsec_scholar.evaluation.metrics import (
    evaluate_answer,
    evaluate_retrieval,
    token_f1,
    unique_evidence,
)
from hardsec_scholar.evaluation.models import (
    AnswerMetrics,
    EvaluationCategory,
    EvaluationRunResult,
    EvaluationSample,
    EvaluationVariant,
    RetrievalMetrics,
    TokenUsage,
)

__all__ = [
    "DatasetValidationReport",
    "AnswerMetrics",
    "EvaluationCategory",
    "EvaluationRunResult",
    "EvaluationSample",
    "EvaluationVariant",
    "RetrievalMetrics",
    "TokenUsage",
    "evaluate_answer",
    "evaluate_retrieval",
    "load_evaluation_dataset",
    "token_f1",
    "unique_evidence",
    "validate_evaluation_dataset",
]
