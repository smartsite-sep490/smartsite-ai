"""Explicit two-artifact composition: person localization separate from PPE objects."""

import hashlib
import json

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope


class PersonAnchoredPpeDetector:
    """Use reference Person detections and retain every non-Person PPE detection.

    Does not load models, own their resources, identify Workers or merge people.
    The runtime must explicitly load/close and evaluate both configured artifacts.
    """

    def __init__(self, *, ppe: DetectorProtocol, persons: DetectorProtocol) -> None:
        self._ppe, self._persons = ppe, persons

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        ppe = await self._ppe.detect(frame)
        self._require_frame(ppe, frame)
        persons = await self._persons.detect(frame)
        self._require_frame(persons, frame)
        provenance = {
            "composition": "person-anchored-ppe-v1",
            "ppe": [ppe.model_artifact_id, ppe.model_version, ppe.model_sha256],
            "persons": [persons.model_artifact_id, persons.model_version, persons.model_sha256],
        }
        digest = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="person-anchored-ppe",
            model_version="v1",
            model_sha256=digest,
            detections=(
                *(d for d in persons.detections if d.class_name.casefold() == "person"),
                *(d for d in ppe.detections if d.class_name.casefold() != "person"),
            ),
        )

    @staticmethod
    def _require_frame(batch: DetectionBatch, frame: FrameEnvelope) -> None:
        if not isinstance(batch, DetectionBatch) or (
            batch.stream_id,
            batch.session_id,
            batch.camera_external_id,
            batch.captured_at,
            batch.sequence_number,
            batch.frame_width,
            batch.frame_height,
        ) != (
            frame.stream_id,
            frame.session_id,
            frame.camera_external_id,
            frame.captured_at,
            frame.sequence_number,
            frame.width,
            frame.height,
        ):
            raise ValueError("Person/PPE detector returned a different source frame")


__all__ = ["PersonAnchoredPpeDetector"]
