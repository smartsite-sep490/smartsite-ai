"""Provider-neutral, deterministic instructions for annotated evaluation evidence."""

from collections.abc import Sequence
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from smartsite_ai.evaluation.matching import (
    DetectionMatchResult,
    EvaluationPrediction,
)
from smartsite_ai.evaluation.models import (
    MAX_IDENTIFIER_LENGTH,
    MAX_SAFE_INTEGER,
    CanonicalPpeClassName,
    EvaluationBoundingBox,
    GroundTruthObject,
    StrictEvaluationModel,
)

OverlaySource = Literal["GROUND_TRUTH", "PREDICTION"]
OverlayMatchStatus = Literal["TP", "FP", "FN"]


class OverlayColor(StrictEvaluationModel):
    """An RGB color that does not depend on a renderer's channel convention."""

    red: int = Field(ge=0, le=255)
    green: int = Field(ge=0, le=255)
    blue: int = Field(ge=0, le=255)


class OverlayLegendEntry(StrictEvaluationModel):
    """One immutable entry in the evidence overlay legend."""

    key: Literal[
        "PREDICTION_TP",
        "PREDICTION_FP",
        "GROUND_TRUTH_TP",
        "GROUND_TRUTH_FN",
        "PPE_PERSON_RELATION",
        "CONFIRMED_CANDIDATE",
    ]
    label: str = Field(min_length=1, max_length=64)
    color: OverlayColor


PREDICTION_TP_COLOR = OverlayColor(red=46, green=204, blue=113)
PREDICTION_FP_COLOR = OverlayColor(red=231, green=76, blue=60)
GROUND_TRUTH_TP_COLOR = OverlayColor(red=52, green=152, blue=219)
GROUND_TRUTH_FN_COLOR = OverlayColor(red=155, green=89, blue=182)
PPE_PERSON_RELATION_COLOR = OverlayColor(red=241, green=196, blue=15)
CONFIRMED_CANDIDATE_COLOR = OverlayColor(red=230, green=126, blue=34)

FIXED_OVERLAY_LEGEND: tuple[OverlayLegendEntry, ...] = (
    OverlayLegendEntry(
        key="PREDICTION_TP", label="Prediction matched (TP)", color=PREDICTION_TP_COLOR
    ),
    OverlayLegendEntry(
        key="PREDICTION_FP", label="Prediction unmatched (FP)", color=PREDICTION_FP_COLOR
    ),
    OverlayLegendEntry(
        key="GROUND_TRUTH_TP", label="Ground truth matched (TP)", color=GROUND_TRUTH_TP_COLOR
    ),
    OverlayLegendEntry(
        key="GROUND_TRUTH_FN", label="Ground truth missed (FN)", color=GROUND_TRUTH_FN_COLOR
    ),
    OverlayLegendEntry(
        key="PPE_PERSON_RELATION",
        label="PPE to person relation",
        color=PPE_PERSON_RELATION_COLOR,
    ),
    OverlayLegendEntry(
        key="CONFIRMED_CANDIDATE",
        label="Confirmed technical candidate",
        color=CONFIRMED_CANDIDATE_COLOR,
    ),
)


class PredictionOverlayContext(StrictEvaluationModel):
    """Runtime context that is not part of the detector prediction contract."""

    prediction_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    related_person_prediction_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
    )
    track_id: int | None = Field(default=None, ge=0, le=MAX_SAFE_INTEGER)
    confirmed_candidate: bool = False

    @field_validator("prediction_id", "related_person_prediction_id")
    @classmethod
    def reject_blank_or_padded_ids(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("overlay identifiers must be non-blank and unpadded")
        return value


class OverlayBoxInstruction(StrictEvaluationModel):
    """One normalized box and deterministic label to be drawn by an adapter."""

    source: OverlaySource
    object_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    class_name: CanonicalPpeClassName
    bounding_box: EvaluationBoundingBox
    match_status: OverlayMatchStatus
    color: OverlayColor
    label: str = Field(min_length=1, max_length=512)
    related_person_object_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
    )
    track_id: int | None = Field(default=None, ge=0, le=MAX_SAFE_INTEGER)
    confirmed_candidate: bool = False

    @model_validator(mode="after")
    def validate_status_for_source(self) -> Self:
        allowed = {"GROUND_TRUTH": {"TP", "FN"}, "PREDICTION": {"TP", "FP"}}
        if self.match_status not in allowed[self.source]:
            raise ValueError(f"{self.match_status} is invalid for {self.source}")
        return self


class OverlayRelationInstruction(StrictEvaluationModel):
    """A PPE-to-person line represented by renderer-independent object keys."""

    source_key: str = Field(min_length=1, max_length=2 * MAX_IDENTIFIER_LENGTH)
    target_key: str = Field(min_length=1, max_length=2 * MAX_IDENTIFIER_LENGTH)
    relation: Literal["PPE_TO_PERSON"] = "PPE_TO_PERSON"
    color: OverlayColor = PPE_PERSON_RELATION_COLOR
    label: Literal["PPE -> Person"] = "PPE -> Person"


class OverlayPlan(StrictEvaluationModel):
    """Complete immutable overlay instructions for exactly one evaluation frame."""

    boxes: tuple[OverlayBoxInstruction, ...]
    relations: tuple[OverlayRelationInstruction, ...]
    legend: tuple[OverlayLegendEntry, ...] = FIXED_OVERLAY_LEGEND
    rendering_excluded_from_timing: Literal[True] = True

    @model_validator(mode="after")
    def require_fixed_legend(self) -> Self:
        if self.legend != FIXED_OVERLAY_LEGEND:
            raise ValueError("overlay plan must use the fixed evidence legend")
        return self


def _validate_input_ids(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[EvaluationPrediction],
) -> tuple[dict[str, GroundTruthObject], dict[str, EvaluationPrediction]]:
    annotations = {item.annotation_id: item for item in ground_truth}
    prediction_map = {item.prediction_id: item for item in predictions}
    if len(annotations) != len(ground_truth):
        raise ValueError("duplicate annotation_id in ground_truth")
    if len(prediction_map) != len(predictions):
        raise ValueError("duplicate prediction_id in predictions")
    for annotation in ground_truth:
        related_id = annotation.related_person_annotation_id
        if related_id is None:
            continue
        related = annotations.get(related_id)
        if related is None or related.class_name != "Person":
            raise ValueError(
                "ground-truth PPE relation must reference a Person annotation in the same frame"
            )
    return annotations, prediction_map


def _validate_match_partition(
    annotations: dict[str, GroundTruthObject],
    predictions: dict[str, EvaluationPrediction],
    result: DetectionMatchResult,
) -> tuple[set[str], set[str]]:
    matched_annotations: set[str] = set()
    matched_predictions: set[str] = set()
    for match in result.matches:
        annotation = annotations.get(match.annotation_id)
        prediction = predictions.get(match.prediction_id)
        if annotation is None or prediction is None:
            raise ValueError("match_result references an unknown annotation or prediction")
        if match.annotation_id in matched_annotations or match.prediction_id in matched_predictions:
            raise ValueError("match_result contains a duplicate match participant")
        if (
            annotation.class_name != prediction.class_name
            or match.class_name != annotation.class_name
        ):
            raise ValueError("match_result class is inconsistent with its objects")
        matched_annotations.add(match.annotation_id)
        matched_predictions.add(match.prediction_id)

    false_positive_ids = [item.prediction_id for item in result.false_positives]
    false_negative_ids = [item.annotation_id for item in result.false_negatives]
    expected_false_positives = set(predictions) - matched_predictions
    expected_false_negatives = set(annotations) - matched_annotations
    if (
        len(false_positive_ids) != len(set(false_positive_ids))
        or len(false_negative_ids) != len(set(false_negative_ids))
        or set(false_positive_ids) != expected_false_positives
        or set(false_negative_ids) != expected_false_negatives
    ):
        raise ValueError("match_result does not partition all frame objects exactly once")
    return matched_annotations, matched_predictions


def _prediction_context_map(
    contexts: Sequence[PredictionOverlayContext],
    predictions: dict[str, EvaluationPrediction],
) -> dict[str, PredictionOverlayContext]:
    context_map = {item.prediction_id: item for item in contexts}
    if len(context_map) != len(contexts):
        raise ValueError("duplicate prediction context")
    for context in contexts:
        prediction = predictions.get(context.prediction_id)
        if prediction is None:
            raise ValueError(f"unknown prediction_id in overlay context: {context.prediction_id}")
        related_id = context.related_person_prediction_id
        if related_id is not None:
            related = predictions.get(related_id)
            if related is None:
                raise ValueError(f"unknown related person prediction: {related_id}")
            if related.class_name != "Person":
                raise ValueError("related_person_prediction_id must reference a Person prediction")
            if prediction.class_name == "Person":
                raise ValueError("Person predictions cannot declare a PPE-to-person relation")
        if context.confirmed_candidate and prediction.class_name not in {
            "NO-Hardhat",
            "NO-Safety Vest",
        }:
            raise ValueError("confirmed candidates require a negative PPE prediction")
        if context.confirmed_candidate and (
            context.track_id is None or context.related_person_prediction_id is None
        ):
            raise ValueError("confirmed candidate requires track_id and a related person")
    return context_map


def _ground_truth_label(item: GroundTruthObject, status: Literal["TP", "FN"]) -> str:
    parts = ["GT", item.class_name, status]
    if item.related_person_annotation_id is not None:
        parts.append(f"person={item.related_person_annotation_id}")
    return " | ".join(parts)


def _prediction_label(
    item: EvaluationPrediction,
    status: Literal["TP", "FP"],
    context: PredictionOverlayContext | None,
) -> str:
    parts = ["PRED", item.class_name, status, f"conf={item.confidence:.3f}"]
    if context is not None:
        if context.track_id is not None:
            parts.append(f"track={context.track_id}")
        if context.related_person_prediction_id is not None:
            parts.append(f"person={context.related_person_prediction_id}")
        if context.confirmed_candidate:
            parts.append("CANDIDATE=CONFIRMED")
    return " | ".join(parts)


def build_overlay_plan(
    ground_truth: Sequence[GroundTruthObject],
    predictions: Sequence[EvaluationPrediction],
    match_result: DetectionMatchResult,
    *,
    prediction_context: Sequence[PredictionOverlayContext] = (),
) -> OverlayPlan:
    """Create a stable evidence plan and reject inconsistent frame-level inputs."""

    annotations, prediction_map = _validate_input_ids(ground_truth, predictions)
    matched_annotations, matched_predictions = _validate_match_partition(
        annotations,
        prediction_map,
        match_result,
    )
    context_map = _prediction_context_map(prediction_context, prediction_map)

    boxes: list[OverlayBoxInstruction] = []
    relations: list[OverlayRelationInstruction] = []
    for annotation_id in sorted(annotations):
        annotation = annotations[annotation_id]
        matched = annotation_id in matched_annotations
        status: Literal["TP", "FN"] = "TP" if matched else "FN"
        boxes.append(
            OverlayBoxInstruction(
                source="GROUND_TRUTH",
                object_id=annotation_id,
                class_name=annotation.class_name,
                bounding_box=annotation.bounding_box,
                match_status=status,
                color=GROUND_TRUTH_TP_COLOR if matched else GROUND_TRUTH_FN_COLOR,
                label=_ground_truth_label(annotation, status),
                **(
                    {"related_person_object_id": annotation.related_person_annotation_id}
                    if annotation.related_person_annotation_id is not None
                    else {}
                ),
            )
        )
        if annotation.related_person_annotation_id is not None:
            relations.append(
                OverlayRelationInstruction(
                    source_key=f"GROUND_TRUTH:{annotation_id}",
                    target_key=f"GROUND_TRUTH:{annotation.related_person_annotation_id}",
                )
            )

    for prediction_id in sorted(prediction_map):
        prediction = prediction_map[prediction_id]
        context = context_map.get(prediction_id)
        matched = prediction_id in matched_predictions
        status: Literal["TP", "FP"] = "TP" if matched else "FP"
        optional_fields: dict[str, object] = {}
        if context is not None:
            if context.related_person_prediction_id is not None:
                optional_fields["related_person_object_id"] = context.related_person_prediction_id
                relations.append(
                    OverlayRelationInstruction(
                        source_key=f"PREDICTION:{prediction_id}",
                        target_key=f"PREDICTION:{context.related_person_prediction_id}",
                    )
                )
            if context.track_id is not None:
                optional_fields["track_id"] = context.track_id
            optional_fields["confirmed_candidate"] = context.confirmed_candidate
        boxes.append(
            OverlayBoxInstruction(
                source="PREDICTION",
                object_id=prediction_id,
                class_name=prediction.class_name,
                bounding_box=prediction.bounding_box,
                match_status=status,
                color=PREDICTION_TP_COLOR if matched else PREDICTION_FP_COLOR,
                label=_prediction_label(prediction, status, context),
                **optional_fields,
            )
        )

    relations.sort(key=lambda item: (item.source_key, item.target_key))
    return OverlayPlan(boxes=tuple(boxes), relations=tuple(relations))


class OverlayRenderer[FrameT](Protocol):
    """Explicit provider boundary; implementations may use OpenCV or another backend."""

    def render(self, frame: FrameT, plan: OverlayPlan) -> FrameT: ...


def render_overlay[FrameT](
    frame: FrameT,
    plan: OverlayPlan,
    *,
    renderer: OverlayRenderer[FrameT],
) -> FrameT:
    """Render outside the measured inference path through an explicitly supplied adapter."""

    return renderer.render(frame, plan)


__all__ = [
    "FIXED_OVERLAY_LEGEND",
    "OverlayBoxInstruction",
    "OverlayColor",
    "OverlayLegendEntry",
    "OverlayPlan",
    "OverlayRelationInstruction",
    "OverlayRenderer",
    "PredictionOverlayContext",
    "build_overlay_plan",
    "render_overlay",
]
