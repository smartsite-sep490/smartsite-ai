import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.detection_metrics import (
    DetectionEvaluationFrame,
    compute_detection_metrics,
)
from smartsite_ai.evaluation.matching import EvaluationPrediction
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    EvaluationBoundingBox,
    GroundTruthObject,
)


def _box(x1: float, y1: float, x2: float, y2: float) -> EvaluationBoundingBox:
    return EvaluationBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _gt(annotation_id: str, class_name: str, x1: float = 0.0) -> GroundTruthObject:
    return GroundTruthObject(
        annotationId=annotation_id,
        className=class_name,
        boundingBox=_box(x1, 0.0, x1 + 0.2, 0.2),
    )


def _prediction(
    prediction_id: str,
    class_name: str,
    x1: float = 0.0,
    confidence: float = 0.9,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        prediction_id=prediction_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=_box(x1, 0.0, x1 + 0.2, 0.2),
    )


def _metrics_by_class(report):
    return {item.class_name: item for item in report.per_class}


def test_reports_exact_per_class_micro_and_supported_macro_metrics() -> None:
    frames = (
        DetectionEvaluationFrame(
            frame_id="frame-1",
            ground_truth=(
                _gt("g-person", "Person", 0.0),
                _gt("g-hardhat-1", "Hardhat", 0.3),
                _gt("g-hardhat-2", "Hardhat", 0.6),
                _gt("g-no-hardhat", "NO-Hardhat", 0.0),
            ),
            predictions=(
                _prediction("p-person", "Person", 0.0),
                _prediction("p-hardhat-hit", "Hardhat", 0.3),
                _prediction("p-hardhat-fp", "Hardhat", 0.0),
                _prediction("p-vest-fp", "Safety Vest", 0.7),
            ),
        ),
    )

    report = compute_detection_metrics(
        frames,
        class_names=CANONICAL_PPE_CLASSES,
        iou_threshold=0.5,
    )
    metrics = _metrics_by_class(report)

    assert report.iou_threshold == 0.5
    assert tuple(metrics) == CANONICAL_PPE_CLASSES
    assert (
        metrics["Hardhat"].true_positives,
        metrics["Hardhat"].false_positives,
        metrics["Hardhat"].false_negatives,
        metrics["Hardhat"].support,
    ) == (1, 1, 1, 2)
    assert metrics["Hardhat"].precision.value == 0.5
    assert metrics["Hardhat"].recall.value == 0.5
    assert metrics["Hardhat"].f1.value == 0.5

    assert metrics["NO-Hardhat"].precision.value is None
    assert metrics["NO-Hardhat"].precision.reason == "no predicted positives"
    assert metrics["NO-Hardhat"].recall.value == 0.0
    assert metrics["NO-Hardhat"].f1.value == 0.0

    assert metrics["Safety Vest"].support == 0
    assert metrics["Safety Vest"].precision.value == 0.0
    assert metrics["Safety Vest"].recall.value is None
    assert metrics["Safety Vest"].recall.reason == "no ground-truth support"
    assert metrics["NO-Safety Vest"].precision.value is None
    assert metrics["NO-Safety Vest"].recall.value is None
    assert metrics["NO-Safety Vest"].f1.value is None
    assert metrics["NO-Safety Vest"].f1.reason == "no ground truth or predictions"

    assert report.macro.supported_classes == ("Person", "Hardhat", "NO-Hardhat")
    assert report.macro.excluded_classes == ("Safety Vest", "NO-Safety Vest")
    assert report.macro.precision.value is None
    assert report.macro.precision.reason == "undefined for 1 supported class: NO-Hardhat"
    assert report.macro.recall.value == 0.5
    assert report.macro.f1.value == 0.5

    assert (
        report.micro.true_positives,
        report.micro.false_positives,
        report.micro.false_negatives,
        report.micro.support,
    ) == (2, 2, 2, 4)
    assert report.micro.precision.value == 0.5
    assert report.micro.recall.value == 0.5
    assert report.micro.f1.value == 0.5


def test_empty_evaluation_emits_null_metrics_with_reasons() -> None:
    report = compute_detection_metrics(
        (),
        class_names=CANONICAL_PPE_CLASSES,
        iou_threshold=0.5,
    )

    assert all(item.support == 0 for item in report.per_class)
    assert report.macro.supported_classes == ()
    assert report.macro.excluded_classes == CANONICAL_PPE_CLASSES
    assert report.macro.precision.value is None
    assert report.macro.precision.reason == "no classes have ground-truth support"
    assert report.macro.recall.reason == "no classes have ground-truth support"
    assert report.macro.f1.reason == "no classes have ground-truth support"
    assert report.micro.precision.reason == "no predicted positives"
    assert report.micro.recall.reason == "no ground-truth support"
    assert report.micro.f1.reason == "no ground truth or predictions"


def test_metric_report_is_deeply_immutable() -> None:
    report = compute_detection_metrics(
        (),
        class_names=CANONICAL_PPE_CLASSES,
        iou_threshold=0.5,
    )

    with pytest.raises(ValidationError):
        report.iou_threshold = 0.75
    with pytest.raises(ValidationError):
        report.micro.precision.value = 1.0


def test_requires_exact_canonical_class_set_and_normalizes_output_order() -> None:
    reversed_classes = tuple(reversed(CANONICAL_PPE_CLASSES))
    report = compute_detection_metrics((), class_names=reversed_classes, iou_threshold=0.5)

    assert tuple(item.class_name for item in report.per_class) == CANONICAL_PPE_CLASSES
    assert report.macro.excluded_classes == CANONICAL_PPE_CLASSES

    with pytest.raises(ValueError, match="class_names must contain all canonical PPE classes"):
        compute_detection_metrics((), class_names=(), iou_threshold=0.5)
    with pytest.raises(ValueError, match="class_names must contain all canonical PPE classes"):
        compute_detection_metrics((), class_names=CANONICAL_PPE_CLASSES[:-1], iou_threshold=0.5)
    with pytest.raises(ValueError, match="class_names must not contain duplicates"):
        compute_detection_metrics(
            (), class_names=(*CANONICAL_PPE_CLASSES, "Person"), iou_threshold=0.5
        )
    with pytest.raises(ValueError, match="unsupported canonical class"):
        compute_detection_metrics(
            (), class_names=(*CANONICAL_PPE_CLASSES[:-1], "Helmet"), iou_threshold=0.5
        )


def test_rejects_duplicate_frames() -> None:
    frame = DetectionEvaluationFrame(
        frame_id="frame-1",
        ground_truth=(_gt("g1", "Person"),),
        predictions=(),
    )

    with pytest.raises(ValueError, match="duplicate frame_id"):
        compute_detection_metrics(
            (frame, frame), class_names=CANONICAL_PPE_CLASSES, iou_threshold=0.5
        )


def test_frame_contract_rejects_duplicate_detection_ids_before_aggregation() -> None:
    prediction = _prediction("p1", "Person")
    frame = DetectionEvaluationFrame(
        frame_id="frame-1",
        ground_truth=(),
        predictions=(prediction, prediction),
    )

    with pytest.raises(ValueError, match="duplicate prediction_id"):
        compute_detection_metrics((frame,), class_names=CANONICAL_PPE_CLASSES, iou_threshold=0.5)
