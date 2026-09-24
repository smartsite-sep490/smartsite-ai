"""Evaluation package for YOLO11s PPE monitoring capability gate."""

from smartsite_ai.evaluation.dataset import (
    DatasetValidationError,
    LoadedEvaluationDataset,
    compute_dataset_aggregate_sha256,
    load_evaluation_dataset,
)
from smartsite_ai.evaluation.detection_metrics import (
    ClassDetectionMetrics,
    DetectionCounts,
    DetectionEvaluationFrame,
    DetectionMetricsReport,
    MacroDetectionMetrics,
    MetricValue,
    compute_detection_metrics,
)
from smartsite_ai.evaluation.matching import (
    DetectionMatch,
    DetectionMatchResult,
    EvaluationPrediction,
    match_detections,
)
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    CANONICAL_PPE_CLASSES_SET,
    MAX_ANNOTATIONS_PER_FRAME,
    REQUIRED_SPLIT_NAMES,
    VALID_PPE_ITEMS,
    EvaluationBoundingBox,
    EvaluationFrame,
    EvaluationManifest,
    GroundTruthObject,
    GroundTruthPpeEpisode,
)

__all__ = [
    "CANONICAL_PPE_CLASSES",
    "CANONICAL_PPE_CLASSES_SET",
    "ClassDetectionMetrics",
    "DatasetValidationError",
    "DetectionCounts",
    "DetectionEvaluationFrame",
    "DetectionMatch",
    "DetectionMatchResult",
    "DetectionMetricsReport",
    "EvaluationBoundingBox",
    "EvaluationFrame",
    "EvaluationManifest",
    "EvaluationPrediction",
    "GroundTruthObject",
    "GroundTruthPpeEpisode",
    "LoadedEvaluationDataset",
    "MAX_ANNOTATIONS_PER_FRAME",
    "MacroDetectionMetrics",
    "MetricValue",
    "REQUIRED_SPLIT_NAMES",
    "VALID_PPE_ITEMS",
    "compute_dataset_aggregate_sha256",
    "compute_detection_metrics",
    "load_evaluation_dataset",
    "match_detections",
]
