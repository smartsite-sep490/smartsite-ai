import builtins
import importlib
import sys
from typing import Any

import pytest

from smartsite_ai.evaluation.matching import (
    DetectionMatchResult,
    EvaluationPrediction,
    match_detections,
)
from smartsite_ai.evaluation.models import EvaluationBoundingBox, GroundTruthObject
from smartsite_ai.evaluation.overlay import (
    FIXED_OVERLAY_LEGEND,
    OverlayPlan,
    PredictionOverlayContext,
    build_overlay_plan,
    render_overlay,
)


def _box(x1: float, y1: float, x2: float, y2: float) -> EvaluationBoundingBox:
    return EvaluationBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _ground_truth(
    annotation_id: str,
    class_name: str,
    coordinates: tuple[float, float, float, float],
    *,
    related_person_annotation_id: str | None = None,
) -> GroundTruthObject:
    payload: dict[str, Any] = {
        "annotationId": annotation_id,
        "className": class_name,
        "boundingBox": _box(*coordinates),
    }
    if related_person_annotation_id is not None:
        payload["relatedPersonAnnotationId"] = related_person_annotation_id
    return GroundTruthObject.model_validate(payload)


def _prediction(
    prediction_id: str,
    class_name: str,
    coordinates: tuple[float, float, float, float],
    *,
    confidence: float = 0.9,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        prediction_id=prediction_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=_box(*coordinates),
    )


def test_overlay_module_does_not_import_heavy_rendering_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "smartsite_ai.evaluation.overlay"
    module = sys.modules.pop(module_name)
    package = sys.modules["smartsite_ai.evaluation"]
    real_import = builtins.__import__

    def reject_heavy_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.split(".", maxsplit=1)[0] in {"cv2", "numpy"}:
            raise AssertionError(f"overlay imported heavy dependency: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_heavy_import)
    try:
        importlib.import_module(module_name)
    finally:
        sys.modules[module_name] = module
        package.overlay = module


def test_fixed_legend_is_complete_stable_and_immutable() -> None:
    assert tuple(entry.key for entry in FIXED_OVERLAY_LEGEND) == (
        "PREDICTION_TP",
        "PREDICTION_FP",
        "GROUND_TRUTH_TP",
        "GROUND_TRUTH_FN",
        "PPE_PERSON_RELATION",
        "CONFIRMED_CANDIDATE",
    )
    assert len({entry.color for entry in FIXED_OVERLAY_LEGEND}) == len(FIXED_OVERLAY_LEGEND)

    with pytest.raises(TypeError):
        FIXED_OVERLAY_LEGEND[0] = FIXED_OVERLAY_LEGEND[1]  # type: ignore[index]


def test_plan_contains_boxes_status_relations_track_and_confirmed_candidate() -> None:
    ground_truth = (
        _ground_truth("g-person", "Person", (0.05, 0.05, 0.70, 0.95)),
        _ground_truth(
            "g-hat",
            "NO-Hardhat",
            (0.20, 0.05, 0.45, 0.25),
            related_person_annotation_id="g-person",
        ),
        _ground_truth("g-vest", "Safety Vest", (0.15, 0.35, 0.60, 0.75)),
    )
    predictions = (
        _prediction("p-person", "Person", (0.05, 0.05, 0.70, 0.95)),
        _prediction("p-hat", "NO-Hardhat", (0.20, 0.05, 0.45, 0.25)),
        _prediction("p-extra", "Hardhat", (0.75, 0.05, 0.95, 0.25), confidence=0.7),
    )
    result = match_detections(ground_truth, predictions, iou_threshold=0.5)

    plan = build_overlay_plan(
        ground_truth,
        predictions,
        result,
        prediction_context=(
            PredictionOverlayContext(prediction_id="p-person", track_id=42),
            PredictionOverlayContext(
                prediction_id="p-hat",
                related_person_prediction_id="p-person",
                track_id=42,
                confirmed_candidate=True,
            ),
        ),
    )

    assert plan.legend == FIXED_OVERLAY_LEGEND
    assert plan.rendering_excluded_from_timing is True
    assert [(item.source, item.object_id, item.match_status) for item in plan.boxes] == [
        ("GROUND_TRUTH", "g-hat", "TP"),
        ("GROUND_TRUTH", "g-person", "TP"),
        ("GROUND_TRUTH", "g-vest", "FN"),
        ("PREDICTION", "p-extra", "FP"),
        ("PREDICTION", "p-hat", "TP"),
        ("PREDICTION", "p-person", "TP"),
    ]

    ppe_box = next(item for item in plan.boxes if item.object_id == "p-hat")
    assert ppe_box.track_id == 42
    assert ppe_box.related_person_object_id == "p-person"
    assert ppe_box.confirmed_candidate is True
    assert "track=42" in ppe_box.label
    assert "person=p-person" in ppe_box.label
    assert "CANDIDATE=CONFIRMED" in ppe_box.label

    assert [(item.source_key, item.target_key) for item in plan.relations] == [
        ("GROUND_TRUTH:g-hat", "GROUND_TRUTH:g-person"),
        ("PREDICTION:p-hat", "PREDICTION:p-person"),
    ]


def test_plan_is_deterministic_for_reordered_equivalent_inputs() -> None:
    ground_truth = (
        _ground_truth("g2", "Hardhat", (0.5, 0.0, 1.0, 0.5)),
        _ground_truth("g1", "Person", (0.0, 0.0, 0.5, 1.0)),
    )
    predictions = (
        _prediction("p2", "Hardhat", (0.5, 0.0, 1.0, 0.5)),
        _prediction("p1", "Person", (0.0, 0.0, 0.5, 1.0)),
    )
    result = match_detections(ground_truth, predictions, iou_threshold=0.5)
    reversed_result = match_detections(
        tuple(reversed(ground_truth)),
        tuple(reversed(predictions)),
        iou_threshold=0.5,
    )

    first = build_overlay_plan(
        ground_truth,
        predictions,
        result,
        prediction_context=(
            PredictionOverlayContext(prediction_id="p2"),
            PredictionOverlayContext(prediction_id="p1", track_id=7),
        ),
    )
    second = build_overlay_plan(
        tuple(reversed(ground_truth)),
        tuple(reversed(predictions)),
        reversed_result,
        prediction_context=(
            PredictionOverlayContext(prediction_id="p1", track_id=7),
            PredictionOverlayContext(prediction_id="p2"),
        ),
    )

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_inconsistent_match_result_is_rejected_instead_of_mislabelling_evidence() -> None:
    ground_truth = (_ground_truth("g1", "Person", (0.0, 0.0, 1.0, 1.0)),)
    prediction = _prediction("p1", "Person", (0.0, 0.0, 1.0, 1.0))
    inconsistent = DetectionMatchResult(
        matches=(),
        false_positives=(),
        false_negatives=(),
    )

    with pytest.raises(ValueError, match="match_result does not partition"):
        build_overlay_plan(ground_truth, (prediction,), inconsistent)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (PredictionOverlayContext(prediction_id="missing"), "unknown prediction_id"),
        (
            PredictionOverlayContext(
                prediction_id="ppe",
                related_person_prediction_id="missing-person",
            ),
            "unknown related person",
        ),
    ],
)
def test_context_must_reference_objects_in_the_same_frame(
    context: PredictionOverlayContext,
    message: str,
) -> None:
    prediction = _prediction("ppe", "Hardhat", (0.0, 0.0, 0.5, 0.5))
    result = match_detections((), (prediction,), iou_threshold=0.5)

    with pytest.raises(ValueError, match=message):
        build_overlay_plan((), (prediction,), result, prediction_context=(context,))


def test_relation_target_must_be_a_person_and_context_ids_are_unique() -> None:
    hat = _prediction("hat", "Hardhat", (0.0, 0.0, 0.5, 0.5))
    vest = _prediction("vest", "Safety Vest", (0.5, 0.0, 1.0, 0.5))
    result = match_detections((), (hat, vest), iou_threshold=0.5)

    with pytest.raises(ValueError, match="must reference a Person prediction"):
        build_overlay_plan(
            (),
            (hat, vest),
            result,
            prediction_context=(
                PredictionOverlayContext(
                    prediction_id="hat",
                    related_person_prediction_id="vest",
                ),
            ),
        )
    with pytest.raises(ValueError, match="duplicate prediction context"):
        build_overlay_plan(
            (),
            (hat, vest),
            result,
            prediction_context=(
                PredictionOverlayContext(prediction_id="hat"),
                PredictionOverlayContext(prediction_id="hat"),
            ),
        )


def test_confirmed_candidate_requires_track_and_person_relation() -> None:
    person = _prediction("person", "Person", (0.0, 0.0, 1.0, 1.0))
    missing_hat = _prediction("missing-hat", "NO-Hardhat", (0.1, 0.0, 0.5, 0.3))
    result = match_detections((), (person, missing_hat), iou_threshold=0.5)

    with pytest.raises(ValueError, match="requires track_id and a related person"):
        build_overlay_plan(
            (),
            (person, missing_hat),
            result,
            prediction_context=(
                PredictionOverlayContext(
                    prediction_id="missing-hat",
                    confirmed_candidate=True,
                ),
            ),
        )


def test_ground_truth_relation_must_reference_a_person_in_the_same_frame() -> None:
    orphan = _ground_truth(
        "hat",
        "Hardhat",
        (0.0, 0.0, 0.5, 0.5),
        related_person_annotation_id="missing-person",
    )
    result = match_detections((orphan,), (), iou_threshold=0.5)

    with pytest.raises(ValueError, match="ground-truth PPE relation"):
        build_overlay_plan((orphan,), (), result)


def test_render_overlay_delegates_to_explicit_renderer_without_timing() -> None:
    calls: list[tuple[object, OverlayPlan]] = []

    class FakeRenderer:
        def render(self, frame: object, plan: OverlayPlan) -> object:
            calls.append((frame, plan))
            return {"annotated": frame}

    plan = build_overlay_plan(
        (),
        (),
        match_detections((), (), iou_threshold=0.5),
    )
    frame = object()

    result = render_overlay(frame, plan, renderer=FakeRenderer())

    assert result == {"annotated": frame}
    assert calls == [(frame, plan)]
