from datetime import UTC, datetime
from uuid import UUID

from smartsite_ai.domain.regions import (
    CameraObservationRegionConfiguration,
    CameraRegionConfiguration,
)
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.tracking import IoUPersonTracker

SESSION = UUID("00000000-0000-4000-8000-000000000001")
REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"


def box(x1: float, y1: float, x2: float, y2: float) -> NormalizedBoundingBox:
    return NormalizedBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, coordinate_space="NORMALIZED_0_1")


def detection(
    class_name: str, bounds: tuple[float, float, float, float], confidence: float = 0.9
) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        bounding_box=box(*bounds),
    )


def batch(*detections: NormalizedDetection) -> DetectionBatch:
    return DetectionBatch(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="camera-01",
        captured_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        frame_width=640,
        frame_height=480,
        sequence_number=1,
        model_artifact_id="fake-detector",
        model_version="test",
        model_sha256="a" * 64,
        detections=detections,
    )


def region() -> CameraObservationRegionConfiguration:
    configuration = CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "camera-01",
            "regions": (
                {
                    "regionId": REGION_ID,
                    "geometryVersion": 7,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
                },
            ),
        }
    )
    return configuration.regions[0]


def test_ppe_pipeline_associates_items_with_the_correct_person() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("person", (0.55, 0.10, 0.85, 0.80), 0.9),
            detection("helmet", (0.62, 0.13, 0.70, 0.22), 0.95),
        )
    )

    observations = PpePipeline().process(tracked, region())

    present = [observation for observation in observations if observation.status == "PRESENT"]
    assert [(observation.track_id, observation.ppe_item) for observation in present] == [
        (2, "HARD_HAT")
    ]
    assert all(observation.region_id == REGION_ID for observation in observations)
    assert all(observation.geometry_version == 7 for observation in observations)


def test_ppe_pipeline_emits_missing_only_for_an_observable_person() -> None:
    visible = IoUPersonTracker().update(batch(detection("person", (0.20, 0.10, 0.40, 0.80))))
    clipped = IoUPersonTracker().update(batch(detection("person", (0.00, 0.10, 0.20, 0.80))))

    visible_observations = PpePipeline().process(visible, region())
    clipped_observations = PpePipeline().process(clipped, region())

    assert [(observation.ppe_item, observation.status) for observation in visible_observations] == [
        ("HARD_HAT", "MISSING"),
        ("SAFETY_VEST", "MISSING"),
    ]
    assert clipped_observations == ()
