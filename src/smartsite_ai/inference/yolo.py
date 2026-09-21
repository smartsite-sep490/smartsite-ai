"""Provider-neutral YOLO row normalization for the detector boundary."""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from math import isfinite
from typing import Protocol, runtime_checkable

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

_MAX_DETECTIONS = 1_024
_RAW_RESULT_LIMIT = _MAX_DETECTIONS + 1


class DetectorUnavailableError(RuntimeError):
    """The detector cannot produce a result for the supplied frame."""


class InferenceResultError(ValueError):
    """The runner produced a result outside the normalized detection contract."""


@dataclass(frozen=True, slots=True)
class RawYoloDetection:
    """One provider-neutral YOLO result in source-frame pixel coordinates."""

    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@runtime_checkable
class YoloRunnerProtocol(Protocol):
    """Synchronously predict raw YOLO rows for one packed BGR frame."""

    def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
        """Return source-frame pixel-space detections without normalization."""
        ...


class Yolo11Detector:
    """Normalize injected YOLO runner results into the locked detection contract."""

    def __init__(self, artifact: VerifiedModelArtifact, runner: YoloRunnerProtocol) -> None:
        self._artifact = artifact
        self._runner = runner
        self._class_map = dict(artifact.class_map)

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        """Run one prediction without blocking the event loop and normalize its rows."""
        if frame.pixel_format != "BGR24":
            raise DetectorUnavailableError("YOLO runner requires BGR24 frames")

        try:
            raw_detections = await asyncio.to_thread(self._predict_rows, frame)
        except Exception as error:
            raise DetectorUnavailableError("YOLO runner prediction failed") from error

        if len(raw_detections) == _RAW_RESULT_LIMIT:
            raise InferenceResultError(
                f"YOLO result contains more than {_MAX_DETECTIONS:,} raw detections"
            )

        detections = tuple(self._normalize_detection(row, frame) for row in raw_detections)
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id=self._artifact.artifact_id,
            model_version=self._artifact.version,
            model_sha256=self._artifact.actual_sha256,
            detections=detections,
        )

    def _predict_rows(self, frame: FrameEnvelope) -> tuple[RawYoloDetection, ...]:
        return tuple(islice(self._runner.predict(frame), _RAW_RESULT_LIMIT))

    def _normalize_detection(
        self, raw_detection: RawYoloDetection, frame: FrameEnvelope
    ) -> NormalizedDetection:
        if not isinstance(raw_detection.class_id, int) or isinstance(raw_detection.class_id, bool):
            raise InferenceResultError("YOLO result class ID must be an integer")
        if raw_detection.class_id not in self._class_map:
            raise InferenceResultError(
                f"YOLO result contains unknown class ID {raw_detection.class_id}"
            )

        values = (
            raw_detection.confidence,
            raw_detection.x1,
            raw_detection.y1,
            raw_detection.x2,
            raw_detection.y2,
        )
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            raise InferenceResultError("YOLO result coordinates and confidence must be numeric")
        if not all(isfinite(value) for value in values):
            raise InferenceResultError("YOLO result coordinates and confidence must be finite")
        if not 0.0 <= raw_detection.confidence <= 1.0:
            raise InferenceResultError("YOLO result confidence must be within [0, 1]")

        x1 = min(max(raw_detection.x1, 0.0), float(frame.width))
        y1 = min(max(raw_detection.y1, 0.0), float(frame.height))
        x2 = min(max(raw_detection.x2, 0.0), float(frame.width))
        y2 = min(max(raw_detection.y2, 0.0), float(frame.height))
        if x1 >= x2 or y1 >= y2:
            raise InferenceResultError("YOLO result bounding box must retain positive area")

        return NormalizedDetection(
            class_id=raw_detection.class_id,
            class_name=self._class_map[raw_detection.class_id],
            confidence=raw_detection.confidence,
            bounding_box=NormalizedBoundingBox(
                x1=x1 / frame.width,
                y1=y1 / frame.height,
                x2=x2 / frame.width,
                y2=y2 / frame.height,
            ),
        )
