import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.matching import EvaluationPrediction, match_detections
from smartsite_ai.evaluation.models import EvaluationBoundingBox, GroundTruthObject


def _box(x1: float, y1: float, x2: float, y2: float) -> EvaluationBoundingBox:
    return EvaluationBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _gt(
    annotation_id: str,
    class_name: str,
    coordinates: tuple[float, float, float, float],
) -> GroundTruthObject:
    return GroundTruthObject(
        annotationId=annotation_id,
        className=class_name,
        boundingBox=_box(*coordinates),
    )


def _prediction(
    prediction_id: str,
    class_name: str,
    confidence: float,
    coordinates: tuple[float, float, float, float],
) -> EvaluationPrediction:
    return EvaluationPrediction(
        prediction_id=prediction_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=_box(*coordinates),
    )


def test_iou_at_threshold_matches_and_just_below_does_not() -> None:
    ground_truth = (_gt("g1", "Hardhat", (0.0, 0.0, 1.0, 1.0)),)

    at_threshold = match_detections(
        ground_truth,
        (_prediction("p1", "Hardhat", 0.9, (0.0, 0.0, 0.5, 1.0)),),
        iou_threshold=0.5,
    )
    below_threshold = match_detections(
        ground_truth,
        (_prediction("p2", "Hardhat", 0.9, (0.0, 0.0, 0.499, 1.0)),),
        iou_threshold=0.5,
    )

    assert [(item.annotation_id, item.prediction_id) for item in at_threshold.matches] == [
        ("g1", "p1")
    ]
    assert below_threshold.matches == ()
    assert [item.annotation_id for item in below_threshold.false_negatives] == ["g1"]
    assert [item.prediction_id for item in below_threshold.false_positives] == ["p2"]


def test_one_prediction_can_match_only_one_ground_truth() -> None:
    result = match_detections(
        (
            _gt("g2", "Person", (0.0, 0.0, 1.0, 1.0)),
            _gt("g1", "Person", (0.0, 0.0, 1.0, 1.0)),
        ),
        (_prediction("p1", "Person", 0.8, (0.0, 0.0, 1.0, 1.0)),),
        iou_threshold=0.5,
    )

    assert [(item.annotation_id, item.prediction_id) for item in result.matches] == [("g1", "p1")]
    assert [item.annotation_id for item in result.false_negatives] == ["g2"]


def test_duplicate_predictions_leave_lower_confidence_as_false_positive() -> None:
    result = match_detections(
        (_gt("g1", "Hardhat", (0.0, 0.0, 0.5, 0.5)),),
        (
            _prediction("p1", "Hardhat", 0.9, (0.0, 0.0, 0.5, 0.5)),
            _prediction("p2", "Hardhat", 0.8, (0.0, 0.0, 0.5, 0.5)),
        ),
        iou_threshold=0.5,
    )

    assert [(item.annotation_id, item.prediction_id) for item in result.matches] == [("g1", "p1")]
    assert [item.prediction_id for item in result.false_positives] == ["p2"]


def test_empty_sides_are_reported_without_special_cases() -> None:
    only_predictions = match_detections(
        (),
        (_prediction("p1", "Person", 0.5, (0.0, 0.0, 1.0, 1.0)),),
        iou_threshold=0.5,
    )
    only_ground_truth = match_detections(
        (_gt("g1", "Person", (0.0, 0.0, 1.0, 1.0)),),
        (),
        iou_threshold=0.5,
    )

    assert only_predictions.matches == ()
    assert [item.prediction_id for item in only_predictions.false_positives] == ["p1"]
    assert [item.annotation_id for item in only_ground_truth.false_negatives] == ["g1"]


def test_cross_class_boxes_never_match() -> None:
    result = match_detections(
        (_gt("g1", "Hardhat", (0.0, 0.0, 1.0, 1.0)),),
        (_prediction("p1", "NO-Hardhat", 0.99, (0.0, 0.0, 1.0, 1.0)),),
        iou_threshold=0.5,
    )

    assert result.matches == ()
    assert [item.annotation_id for item in result.false_negatives] == ["g1"]
    assert [item.prediction_id for item in result.false_positives] == ["p1"]


def test_candidate_order_is_iou_then_confidence_then_stable_ids() -> None:
    iou_wins = match_detections(
        (_gt("g", "Person", (0.0, 0.0, 1.0, 1.0)),),
        (
            _prediction("higher-confidence", "Person", 0.99, (0.0, 0.0, 0.8, 1.0)),
            _prediction("higher-iou", "Person", 0.50, (0.0, 0.0, 1.0, 1.0)),
        ),
        iou_threshold=0.5,
    )
    stable_ids = match_detections(
        (
            _gt("g2", "Person", (0.0, 0.0, 1.0, 1.0)),
            _gt("g1", "Person", (0.0, 0.0, 1.0, 1.0)),
        ),
        (
            _prediction("p2", "Person", 0.8, (0.0, 0.0, 1.0, 1.0)),
            _prediction("p1", "Person", 0.8, (0.0, 0.0, 1.0, 1.0)),
        ),
        iou_threshold=0.5,
    )

    assert iou_wins.matches[0].prediction_id == "higher-iou"
    assert [(item.annotation_id, item.prediction_id) for item in stable_ids.matches] == [
        ("g1", "p1"),
        ("g2", "p2"),
    ]


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_iou_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections((), (), iou_threshold=threshold)


def test_prediction_contract_rejects_unknown_class_and_padded_identifier() -> None:
    with pytest.raises(ValidationError):
        _prediction("p1", "Helmet", 0.9, (0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValidationError):
        _prediction(" p1", "Hardhat", 0.9, (0.0, 0.0, 1.0, 1.0))


def test_duplicate_input_identifiers_are_rejected() -> None:
    prediction = _prediction("p1", "Person", 0.8, (0.0, 0.0, 1.0, 1.0))
    annotation = _gt("g1", "Person", (0.0, 0.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="duplicate prediction_id"):
        match_detections((), (prediction, prediction), iou_threshold=0.5)
    with pytest.raises(ValueError, match="duplicate annotation_id"):
        match_detections((annotation, annotation), (), iou_threshold=0.5)
