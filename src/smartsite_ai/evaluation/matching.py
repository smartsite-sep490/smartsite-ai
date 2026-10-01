"""Pure deterministic one-to-one matching for PPE detection evaluation."""

import math
from collections.abc import Sequence
from typing import Self

from pydantic import Field, field_validator, model_validator

from smartsite_ai.evaluation.models import (
    MAX_IDENTIFIER_LENGTH,
    CanonicalPpeClassName,
    EvaluationBoundingBox,
    GroundTruthObject,
    StrictEvaluationModel,
)


class EvaluationPrediction(StrictEvaluationModel):
    """One detector prediction with a stable evaluation-local identifier."""

    prediction_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    class_name: CanonicalPpeClassName
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: EvaluationBoundingBox

    @field_validator("prediction_id")
    @classmethod
    def validate_prediction_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("prediction_id must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_finite_confidence(self) -> Self:
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        return self


class DetectionMatch(StrictEvaluationModel):
    """One selected ground-truth/prediction pair."""

    annotation_id: str
    prediction_id: str
    class_name: CanonicalPpeClassName
    iou: float = Field(ge=0.0, le=1.0)


class DetectionMatchResult(StrictEvaluationModel):
    """Immutable selected matches and unmatched inputs."""

    matches: tuple[DetectionMatch, ...]
    false_positives: tuple[EvaluationPrediction, ...]
    false_negatives: tuple[GroundTruthObject, ...]


def _intersection_over_union(
    left: EvaluationBoundingBox,
    right: EvaluationBoundingBox,
) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection_area = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union_area = left_area + right_area - intersection_area
    return intersection_area / union_area


def _reject_duplicate_ids(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[EvaluationPrediction],
) -> None:
    annotation_ids = [item.annotation_id for item in ground_truth]
    prediction_ids = [item.prediction_id for item in predictions]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate annotation_id in ground_truth")
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("duplicate prediction_id in predictions")


def match_detections(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[EvaluationPrediction],
    *,
    iou_threshold: float,
) -> DetectionMatchResult:
    """Match same-class boxes greedily using the locked deterministic ordering."""

    if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in the interval (0, 1]")
    _reject_duplicate_ids(ground_truth, predictions)

    candidates: list[tuple[float, float, str, str, GroundTruthObject, EvaluationPrediction]] = []
    for annotation in ground_truth:
        for prediction in predictions:
            if annotation.class_name != prediction.class_name:
                continue
            iou = _intersection_over_union(annotation.bounding_box, prediction.bounding_box)
            if iou >= iou_threshold:
                candidates.append(
                    (
                        -iou,
                        -prediction.confidence,
                        prediction.prediction_id,
                        annotation.annotation_id,
                        annotation,
                        prediction,
                    )
                )

    candidates.sort(key=lambda candidate: candidate[:4])
    matched_annotations: set[str] = set()
    matched_predictions: set[str] = set()
    matches: list[DetectionMatch] = []
    for negative_iou, _, _, _, annotation, prediction in candidates:
        if (
            annotation.annotation_id in matched_annotations
            or prediction.prediction_id in matched_predictions
        ):
            continue
        matched_annotations.add(annotation.annotation_id)
        matched_predictions.add(prediction.prediction_id)
        matches.append(
            DetectionMatch(
                annotation_id=annotation.annotation_id,
                prediction_id=prediction.prediction_id,
                class_name=annotation.class_name,
                iou=-negative_iou,
            )
        )

    return DetectionMatchResult(
        matches=tuple(matches),
        false_positives=tuple(
            sorted(
                (item for item in predictions if item.prediction_id not in matched_predictions),
                key=lambda item: item.prediction_id,
            )
        ),
        false_negatives=tuple(
            sorted(
                (item for item in ground_truth if item.annotation_id not in matched_annotations),
                key=lambda item: item.annotation_id,
            )
        ),
    )


__all__ = [
    "DetectionMatch",
    "DetectionMatchResult",
    "EvaluationPrediction",
    "match_detections",
]
