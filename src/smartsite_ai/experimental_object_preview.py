"""Same-frame unified8 raw objects; never wearer states or safety decisions."""

from typing import Any

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.ppe_profiles import UNIFIED_PPE_15_PROFILE
from smartsite_ai.ingestion.envelope import FrameEnvelope


def unified_object_preview(batch: DetectionBatch, frame: FrameEnvelope) -> dict[str, Any]:
    """Project a validated detection batch without losing its JPEG source lineage."""
    if (
        batch.camera_external_id != frame.camera_external_id
        or batch.stream_id != frame.stream_id
        or batch.session_id != frame.session_id
        or batch.sequence_number != frame.sequence_number
        or batch.captured_at != frame.captured_at
        or (batch.frame_width, batch.frame_height) != (frame.width, frame.height)
    ):
        raise ValueError("experimental object/frame lineage mismatch")
    expected = dict(UNIFIED_PPE_15_PROFILE.class_map)
    detections = []
    for detection in batch.detections:
        if expected.get(detection.class_id) != detection.class_name:
            raise ValueError("experimental object class taxonomy mismatch")
        box = detection.bounding_box
        detections.append(
            {
                "classId": detection.class_id,
                "className": detection.class_name,
                "confidence": detection.confidence,
                "boundingBox": {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "coordinateSpace": "NORMALIZED_0_1",
                },
            }
        )
    return {
        "modelFamily": "yolo11s",
        "experimental": True,
        "scope": "OBJECT_DETECTION_ONLY",
        "artifactId": batch.model_artifact_id,
        "artifactSha256": batch.model_sha256,
        "detections": detections,
    }
