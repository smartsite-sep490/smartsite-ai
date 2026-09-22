from datetime import UTC, datetime
from uuid import UUID

import pytest

from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.tracking import IoUPersonTracker

SESSION = UUID("00000000-0000-4000-8000-000000000001")
REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"


def box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, coordinate_space="NORMALIZED_0_1")


def person(bounds: tuple[float, float, float, float]) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=0,
        class_name="person",
        confidence=0.91,
        bounding_box=box(*bounds),
    )


def batch(sequence_number: int, person_detection: NormalizedDetection) -> DetectionBatch:
    return DetectionBatch(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="camera-01",
        captured_at=datetime(2026, 9, 21, 12, 0, sequence_number, tzinfo=UTC),
        frame_width=640,
        frame_height=480,
        sequence_number=sequence_number,
        model_artifact_id="fake-detector",
        model_version="test",
        model_sha256="a" * 64,
        detections=(person_detection,),
    )


def configuration(geometry_version: int = 7) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": geometry_version,
            "cameraExternalId": "camera-01",
            "regions": (
                {
                    "regionId": REGION_ID,
                    "geometryVersion": geometry_version,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8))},
                },
            ),
        }
    )


def test_zone_pipeline_emits_only_on_outside_to_inside_transition() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()

    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    inside = tracker.update(batch(2, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(outside, configuration()) == ()
    entries = zones.process(inside, configuration())

    assert len(entries) == 1
    assert entries[0].track_id == 1
    assert entries[0].region_id == REGION_ID
    assert entries[0].geometry_version == 7


def test_zone_pipeline_does_not_treat_initial_inside_frame_as_an_entry() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    inside = tracker.update(batch(1, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(inside, configuration()) == ()


def test_zone_pipeline_resets_state_when_geometry_version_changes() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    inside = tracker.update(batch(2, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(outside, configuration()) == ()
    assert zones.process(inside, configuration())
    moved_back_outside = tracker.update(batch(3, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(moved_back_outside, configuration(geometry_version=8)) == ()


def test_zone_pipeline_rejects_wrong_camera_and_out_of_order_frames() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    tracked = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    wrong_camera = configuration().model_copy(update={"camera_external_id": "camera-02"})

    with pytest.raises(ValueError, match="does not match"):
        zones.process(tracked, wrong_camera)

    zones.process(tracked, configuration())
    with pytest.raises(ValueError, match="strictly increasing"):
        zones.process(tracked, configuration())
