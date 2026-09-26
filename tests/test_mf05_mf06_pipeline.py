from datetime import UTC, datetime
from uuid import UUID

from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.tracking import IoUPersonTracker

SESSION = UUID("00000000-0000-4000-8000-000000000001")
REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
PPE_REGION_ID = "00000000-0000-4000-8000-000000000002"


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


def batch(sequence_number: int, *detections: NormalizedDetection) -> DetectionBatch:
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
        detections=detections,
    )


def configuration() -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "camera-01",
            "regions": (
                {
                    "regionId": PPE_REGION_ID,
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
                },
                {
                    "regionId": REGION_ID,
                    "geometryVersion": 7,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8))},
                },
            ),
        }
    )


def make_pipeline() -> Mf05Mf06Pipeline:
    return Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=PPE_REGION_ID,
    )


def test_pipeline_builds_technical_observations_without_authorization_decisions() -> None:
    pipeline = make_pipeline()
    first = pipeline.process(
        batch(
            1,
            detection("person", (0.35, 0.20, 0.55, 0.39)),
            detection("helmet", (0.40, 0.21, 0.48, 0.29)),
        ),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000010",
    )
    second = pipeline.process(
        batch(2, detection("person", (0.45, 0.20, 0.55, 0.60))),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000011",
    )
    third = pipeline.process(
        batch(3, detection("person", (0.45, 0.20, 0.55, 0.60))),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000012",
    )
    fourth = pipeline.process(
        batch(4, detection("person", (0.45, 0.20, 0.55, 0.60))),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000013",
    )

    assert first is not None
    assert second is not None
    assert third is not None
    assert fourth is not None
    assert [observation.type for observation in first.observations] == [
        "PERSON",
        "PPE",
    ]
    assert {observation.type for observation in second.observations} == {"PERSON"}
    assert {observation.type for observation in third.observations} == {"PERSON"}
    assert {observation.type for observation in fourth.observations} == {
        "PERSON",
        "ZONE_ENTRY",
    }
    wire = fourth.to_wire_dict()
    assert "allowed" not in wire
    assert "denied" not in wire
    assert "unauthorized" not in wire


def test_pipeline_returns_none_when_a_batch_has_no_person_track() -> None:
    pipeline = make_pipeline()

    assert (
        pipeline.process(
            batch(1, detection("helmet", (0.15, 0.12, 0.22, 0.20))),
            region_configuration=configuration(),
            event_id="00000000-0000-4000-8000-000000000010",
        )
        is None
    )
