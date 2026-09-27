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
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
                },
            ),
        }
    )
    return configuration.regions[0]


def left_half_region() -> CameraObservationRegionConfiguration:
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
                    "polygon": {"coordinates": ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))},
                },
            ),
        }
    )
    return configuration.regions[0]


def test_ppe_pipeline_omits_person_whose_bottom_center_is_outside_region() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.40, 0.10, 0.80, 0.90)),
            detection("NO-Hardhat", (0.55, 0.12, 0.65, 0.25)),
        )
    )
    assert PpePipeline().process(tracked, left_half_region()) == ()


def test_ppe_pipeline_includes_polygon_boundary_bottom_center() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.40, 0.10, 0.60, 0.90)),
            detection("NO-Hardhat", (0.44, 0.12, 0.54, 0.25)),
        )
    )
    observations = PpePipeline().process(tracked, left_half_region())
    assert [(item.ppe_item, item.status) for item in observations] == [("HARD_HAT", "MISSING")]


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


def test_ppe_pipeline_accepts_the_reference_checkpoint_hardhat_class_name() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("Hardhat", (0.14, 0.12, 0.24, 0.25), 0.95),
        )
    )

    observations = PpePipeline().process(tracked, region())

    assert [(observation.ppe_item, observation.status) for observation in observations] == [
        ("HARD_HAT", "PRESENT"),
    ]


def test_ppe_pipeline_requires_an_explicit_negative_class_for_missing() -> None:
    visible = IoUPersonTracker().update(
        batch(
            detection("person", (0.20, 0.10, 0.40, 0.80)),
            detection("NO-Hardhat", (0.24, 0.12, 0.34, 0.25)),
            detection("NO-Safety Vest", (0.23, 0.34, 0.38, 0.70)),
        )
    )
    clipped = IoUPersonTracker().update(batch(detection("person", (0.00, 0.10, 0.20, 0.80))))

    visible_observations = PpePipeline().process(visible, region())
    clipped_observations = PpePipeline().process(clipped, region())

    assert [(observation.ppe_item, observation.status) for observation in visible_observations] == [
        ("HARD_HAT", "MISSING"),
        ("SAFETY_VEST", "MISSING"),
    ]
    assert clipped_observations == ()


def test_ppe_pipeline_omits_contradictory_item_evidence() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.40, 0.90)),
            detection("Hardhat", (0.16, 0.12, 0.25, 0.25), 0.95),
            detection("NO-Hardhat", (0.17, 0.12, 0.26, 0.25), 0.94),
            detection("Safety Vest", (0.15, 0.35, 0.35, 0.70), 0.90),
        )
    )
    observations = PpePipeline().process(tracked, region())
    assert [(item.ppe_item, item.status) for item in observations] == [("SAFETY_VEST", "PRESENT")]


def test_ppe_pipeline_suppresses_missing_fallback_for_conflicted_evidence() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.40, 0.90)),
            detection("Hardhat", (0.16, 0.12, 0.25, 0.25), 0.95),
            detection("NO-Hardhat", (0.17, 0.12, 0.26, 0.25), 0.94),
            detection("Safety Vest", (0.15, 0.35, 0.35, 0.70), 0.90),
        )
    )
    pipeline = PpePipeline(emit_missing=True)
    observations = pipeline.process(tracked, region())
    assert [(item.ppe_item, item.status) for item in observations] == [("SAFETY_VEST", "PRESENT")]
