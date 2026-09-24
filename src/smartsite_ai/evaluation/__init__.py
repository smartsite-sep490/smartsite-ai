"""Evaluation package for YOLO11s PPE monitoring capability gate."""

from smartsite_ai.evaluation.dataset import (
    DatasetValidationError,
    LoadedEvaluationDataset,
    compute_dataset_aggregate_sha256,
    load_evaluation_dataset,
)
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    EvaluationBoundingBox,
    EvaluationFrame,
    EvaluationManifest,
    GroundTruthObject,
    GroundTruthPpeEpisode,
)

__all__ = [
    "CANONICAL_PPE_CLASSES",
    "DatasetValidationError",
    "EvaluationBoundingBox",
    "EvaluationFrame",
    "EvaluationManifest",
    "GroundTruthObject",
    "GroundTruthPpeEpisode",
    "LoadedEvaluationDataset",
    "compute_dataset_aggregate_sha256",
    "load_evaluation_dataset",
]
