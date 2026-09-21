"""Explicit-lifecycle Ultralytics adapter for packed BGR24 frames."""

from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    InferenceResultError,
    RawYoloDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

_MAX_RAW_DETECTIONS = 1_024
_RAW_RESULT_LIMIT = _MAX_RAW_DETECTIONS + 1

ModelFactory = Callable[[Path], object]


class UltralyticsYoloRunner:
    """Run a verified local Ultralytics model only after explicit loading."""

    def __init__(self, *, model_factory: ModelFactory | None = None) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._model: object | None = None
        self._artifact: VerifiedModelArtifact | None = None

    def load(self, artifact: VerifiedModelArtifact) -> None:
        """Construct the model from an already verified local artifact."""
        if self._model is not None:
            raise DetectorUnavailableError("Ultralytics runner is already loaded")

        try:
            model = self._model_factory(artifact.resolved_path)
        except Exception as error:
            raise DetectorUnavailableError("Ultralytics model construction failed") from error

        self._model = model
        self._artifact = artifact

    def predict(self, frame: FrameEnvelope) -> tuple[RawYoloDetection, ...]:
        """Predict provider-neutral source-frame rows for one BGR24 frame."""
        model = self._model
        artifact = self._artifact
        if model is None or artifact is None:
            raise DetectorUnavailableError("Ultralytics runner is not loaded")
        if frame.pixel_format != "BGR24":
            raise DetectorUnavailableError("Ultralytics runner requires BGR24 frames")

        image = _bgr_image(frame)
        try:
            results = model.predict(
                source=image,
                conf=artifact.confidence_threshold,
                iou=artifact.iou_threshold,
                imgsz=artifact.image_size,
                device=artifact.device,
                verbose=False,
            )
        except Exception as error:
            raise DetectorUnavailableError("Ultralytics provider prediction failed") from error

        try:
            return _translate_first_result(results, artifact)
        except InferenceResultError:
            raise
        except Exception as error:
            raise DetectorUnavailableError(
                "Ultralytics provider returned an invalid result"
            ) from error

    def close(self) -> None:
        """Drop references to the model and its verified artifact."""
        self._model = None
        self._artifact = None


def _default_model_factory(path: Path) -> object:
    """Create the provider model without allowing implicit import-time loading."""
    from ultralytics import YOLO

    return YOLO(path)


def _bgr_image(frame: FrameEnvelope) -> object:
    """Make the read-only provider image view from immutable packed BGR bytes."""
    import numpy as np

    return np.frombuffer(frame.buffer, dtype=np.uint8).reshape((frame.height, frame.width, 3))


def _translate_first_result(
    results: object, artifact: VerifiedModelArtifact
) -> tuple[RawYoloDetection, ...]:
    first_result = next(iter(results), None)
    if first_result is None:
        return ()

    boxes = first_result.boxes
    if boxes is None:
        return ()
    names = first_result.names
    class_map = dict(artifact.class_map)
    rows: list[RawYoloDetection] = []
    for coordinates, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
    ):
        if len(rows) == _RAW_RESULT_LIMIT:
            break
        normalized_class_id = _class_id(class_id)
        _verify_class_name(names, normalized_class_id, class_map)
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise InferenceResultError("Ultralytics provider box must contain four coordinates")
        x1, y1, x2, y2 = (_finite_float(value, "coordinate") for value in coordinates)
        rows.append(
            RawYoloDetection(
                class_id=normalized_class_id,
                confidence=_finite_float(confidence, "confidence"),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return tuple(rows)


def _class_id(value: object) -> int:
    if isinstance(value, bool):
        raise InferenceResultError("Ultralytics provider class ID must be an integer")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceResultError("Ultralytics provider class ID must be an integer") from error
    if not isfinite(numeric_value) or not numeric_value.is_integer():
        raise InferenceResultError("Ultralytics provider class ID must be an integer")
    return int(numeric_value)


def _finite_float(value: object, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceResultError(f"Ultralytics provider {label} must be numeric") from error
    if not isfinite(numeric_value):
        raise InferenceResultError(f"Ultralytics provider {label} must be finite")
    return numeric_value


def _verify_class_name(names: object, class_id: int, class_map: dict[int, str]) -> None:
    if not isinstance(names, Mapping):
        raise InferenceResultError("Ultralytics provider class names must be a mapping")
    expected_name = class_map.get(class_id)
    provider_name = names.get(class_id)
    if expected_name is None or provider_name != expected_name:
        raise InferenceResultError(
            f"Ultralytics provider class name disagrees for class ID {class_id}"
        )
