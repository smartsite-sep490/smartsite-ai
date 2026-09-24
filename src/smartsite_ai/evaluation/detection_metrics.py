"""Deterministic per-class, macro, and micro PPE detection metrics."""

import math
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartsite_ai.evaluation.matching import EvaluationPrediction, match_detections
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    CANONICAL_PPE_CLASSES_SET,
    MAX_IDENTIFIER_LENGTH,
    CanonicalPpeClassName,
    GroundTruthObject,
)


class ImmutableMetricsModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class DetectionEvaluationFrame(ImmutableMetricsModel):
    """Ground truth and predictions belonging to one evaluation frame."""

    frame_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    ground_truth: tuple[GroundTruthObject, ...]
    predictions: tuple[EvaluationPrediction, ...]

    @field_validator("frame_id")
    @classmethod
    def validate_frame_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("frame_id must be non-blank and unpadded")
        return value


class MetricValue(ImmutableMetricsModel):
    """A metric value, or an explicit explanation of why it is undefined."""

    value: float | None
    reason: str | None

    @model_validator(mode="after")
    def validate_value_or_reason(self) -> Self:
        if (self.value is None) == (self.reason is None):
            raise ValueError("exactly one of value or reason must be set")
        return self


class DetectionCounts(ImmutableMetricsModel):
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    support: int = Field(ge=0)
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue

    @model_validator(mode="after")
    def validate_support(self) -> Self:
        if self.support != self.true_positives + self.false_negatives:
            raise ValueError("support must equal true_positives + false_negatives")
        return self


class ClassDetectionMetrics(DetectionCounts):
    class_name: CanonicalPpeClassName


class MacroDetectionMetrics(ImmutableMetricsModel):
    supported_classes: tuple[CanonicalPpeClassName, ...]
    excluded_classes: tuple[CanonicalPpeClassName, ...]
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue


class DetectionMetricsReport(ImmutableMetricsModel):
    iou_threshold: float = Field(gt=0.0, le=1.0)
    per_class: tuple[ClassDetectionMetrics, ...]
    macro: MacroDetectionMetrics
    micro: DetectionCounts


def _defined(value: float) -> MetricValue:
    return MetricValue(value=value, reason=None)


def _undefined(reason: str) -> MetricValue:
    return MetricValue(value=None, reason=reason)


def _scores(
    true_positives: int, false_positives: int, false_negatives: int
) -> tuple[MetricValue, MetricValue, MetricValue]:
    predicted_positives = true_positives + false_positives
    support = true_positives + false_negatives
    f1_denominator = 2 * true_positives + false_positives + false_negatives

    precision = (
        _defined(true_positives / predicted_positives)
        if predicted_positives
        else _undefined("no predicted positives")
    )
    recall = (
        _defined(true_positives / support) if support else _undefined("no ground-truth support")
    )
    f1 = (
        _defined((2 * true_positives) / f1_denominator)
        if f1_denominator
        else _undefined("no ground truth or predictions")
    )
    return precision, recall, f1


def _counts(*, true_positives: int, false_positives: int, false_negatives: int) -> DetectionCounts:
    precision, recall, f1 = _scores(true_positives, false_positives, false_negatives)
    return DetectionCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        support=true_positives + false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _macro_score(
    metrics: Sequence[ClassDetectionMetrics],
    attribute: str,
) -> MetricValue:
    if not metrics:
        return _undefined("no classes have ground-truth support")
    undefined_classes = [
        item.class_name for item in metrics if getattr(item, attribute).value is None
    ]
    if undefined_classes:
        noun = "class" if len(undefined_classes) == 1 else "classes"
        return _undefined(
            f"undefined for {len(undefined_classes)} supported {noun}: "
            + ", ".join(undefined_classes)
        )
    values = [getattr(item, attribute).value for item in metrics]
    return _defined(sum(values) / len(values))


def compute_detection_metrics(
    frames: Sequence[DetectionEvaluationFrame],
    *,
    class_names: Sequence[str],
    iou_threshold: float,
) -> DetectionMetricsReport:
    """Aggregate deterministic matching results without external metric libraries."""

    if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in the interval (0, 1]")
    if len(class_names) != len(set(class_names)):
        raise ValueError("class_names must not contain duplicates")
    unknown_classes = set(class_names) - CANONICAL_PPE_CLASSES_SET
    if unknown_classes:
        raise ValueError(f"unsupported canonical class: {sorted(unknown_classes)}")
    missing_classes = CANONICAL_PPE_CLASSES_SET - set(class_names)
    if missing_classes:
        raise ValueError(
            f"class_names must contain all canonical PPE classes; missing {sorted(missing_classes)}"
        )
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("duplicate frame_id in evaluation frames")

    totals = {
        class_name: {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        for class_name in CANONICAL_PPE_CLASSES
    }
    for frame in frames:
        result = match_detections(
            frame.ground_truth,
            frame.predictions,
            iou_threshold=iou_threshold,
        )
        for item in result.matches:
            totals[item.class_name]["true_positives"] += 1
        for item in result.false_positives:
            totals[item.class_name]["false_positives"] += 1
        for item in result.false_negatives:
            totals[item.class_name]["false_negatives"] += 1

    per_class: list[ClassDetectionMetrics] = []
    for class_name in CANONICAL_PPE_CLASSES:
        class_totals = totals[class_name]
        counts = _counts(**class_totals)
        per_class.append(
            ClassDetectionMetrics(
                class_name=class_name,
                **counts.model_dump(),
            )
        )

    supported = tuple(item for item in per_class if item.support > 0)
    excluded = tuple(item.class_name for item in per_class if item.support == 0)
    macro = MacroDetectionMetrics(
        supported_classes=tuple(item.class_name for item in supported),
        excluded_classes=excluded,
        precision=_macro_score(supported, "precision"),
        recall=_macro_score(supported, "recall"),
        f1=_macro_score(supported, "f1"),
    )
    micro = _counts(
        true_positives=sum(item.true_positives for item in per_class),
        false_positives=sum(item.false_positives for item in per_class),
        false_negatives=sum(item.false_negatives for item in per_class),
    )
    return DetectionMetricsReport(
        iou_threshold=iou_threshold,
        per_class=tuple(per_class),
        macro=macro,
        micro=micro,
    )


__all__ = [
    "ClassDetectionMetrics",
    "DetectionCounts",
    "DetectionEvaluationFrame",
    "DetectionMetricsReport",
    "ImmutableMetricsModel",
    "MacroDetectionMetrics",
    "MetricValue",
    "compute_detection_metrics",
]
