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


def test_disjoint_person_cannot_inherit_another_person_zone_confirmation_at_zero_threshold():
    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(iou_threshold=0),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(entry_confirmation_frames=2),
        ppe_region_id=PPE_REGION_ID,
    )
    events = []
    for sequence, bounds in enumerate(
        [(0.41, 0.10, 0.48, 0.60), (0.68, 0.10, 0.75, 0.60), (0.68, 0.10, 0.75, 0.60)], 1
    ):
        event = pipeline.process(
            batch(sequence, detection("person", bounds)),
            region_configuration=configuration(),
            event_id=f"00000000-0000-4000-8000-{sequence:012d}",
        )
        assert event is not None
        events.append(event)
    assert not any(o.type == "ZONE_ENTRY" for o in events[1].observations)
    people = [[o.track_id for o in event.observations if o.type == "PERSON"] for event in events]
    assert people[0] != people[1] == people[2]
    assert [o.track_id for o in events[2].observations if o.type == "ZONE_ENTRY"] == people[2]


def test_ten_person_scene_keeps_three_missing_ppe_subjects_and_polygon_entries_separate():
    from smartsite_ai.pipelines.ppe_temporal import (
        TemporalPpeCandidateGate,
        filter_event_for_delivery,
    )

    detections = []
    for index in range(10):
        x, y = 0.04 + (index % 5) * 0.18, 0.05 + (index // 5) * 0.45
        detections.extend(
            [
                detection("person", (x, y, x + 0.12, y + 0.35)),
                detection(
                    "no-hardhat" if index < 3 else "hardhat",
                    (x + 0.04, y + 0.02, x + 0.08, y + 0.06),
                ),
                detection(
                    "no-safety vest" if index < 3 else "safety vest",
                    (x + 0.02, y + 0.11, x + 0.10, y + 0.24),
                ),
            ]
        )
    content = configuration().model_dump(mode="python", by_alias=True)
    content["regions"][1]["polygon"]["coordinates"] = (
        (0.0, 0.0),
        (0.57, 0.0),
        (0.57, 0.42),
        (0.0, 0.42),
    )
    config = CameraRegionConfiguration.model_validate(content)
    pipeline, gate = make_pipeline(), TemporalPpeCandidateGate()
    delivered = []
    for sequence in range(1, 5):
        current = batch(sequence, *detections)
        event = pipeline.process(
            current,
            region_configuration=config,
            event_id=f"00000000-0000-4000-8000-{sequence:012d}",
        )
        assert event is not None
        persons = [o.track_id for o in event.observations if o.type == "PERSON"]
        assert len(persons) == len(set(persons)) == 10
        candidates = gate.update(
            stream_id=current.stream_id,
            session_id=current.session_id,
            observed_at=current.captured_at,
            active_track_ids=persons,
            observations=[o for o in event.observations if o.type == "PPE"],
        )
        delivery = filter_event_for_delivery(event, candidates)
        if delivery is not None:
            delivered.append(delivery)
    assert len(delivered) == 1
    missing = [o for o in delivered[0].observations if o.type == "PPE" and o.status == "MISSING"]
    assert {(o.track_id, o.ppe_item) for o in missing} == {
        # Detector batches sort by x then y; the first row's first three are 1/3/5.
        (track, item)
        for track in (1, 3, 5)
        for item in ("HARD_HAT", "SAFETY_VEST")
    }
    entries = [o for o in delivered[0].observations if o.type == "ZONE_ENTRY"]
    assert {o.track_id for o in entries} == {1, 3, 5}
    assert all(o.region_id == REGION_ID and o.geometry_version == 7 for o in entries)
    assert len([o for o in delivered[0].observations if o.type == "PERSON"]) == 10


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
