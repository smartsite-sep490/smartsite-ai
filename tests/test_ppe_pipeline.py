from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

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
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson

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


def overlapping_frame(
    *ppe: NormalizedDetection,
    reverse_people: bool = False,
    swap_track_ids: bool = False,
) -> TrackedFrame:
    left = detection("Person", (0.10, 0.10, 0.60, 0.90), 0.99)
    right = detection("Person", (0.40, 0.10, 0.90, 0.90), 0.70)
    people = (
        TrackedPerson(2 if swap_track_ids else 1, left),
        TrackedPerson(1 if swap_track_ids else 2, right),
    )
    return TrackedFrame(
        batch=batch(left, right, *ppe),
        persons=tuple(reversed(people)) if reverse_people else people,
    )


@pytest.mark.parametrize(
    ("class_name", "ppe_item"),
    [
        ("Hardhat", "HARD_HAT"),
        ("NO-Hardhat", "HARD_HAT"),
        ("Safety Vest", "SAFETY_VEST"),
        ("NO-Safety Vest", "SAFETY_VEST"),
    ],
)
@pytest.mark.parametrize("outside_competitor", [False, True])
@pytest.mark.parametrize("emit_missing", [False, True])
def test_ppe_pipeline_abstains_when_multiple_people_fit_the_same_item(
    class_name: str, ppe_item: str, outside_competitor: bool, emit_missing: bool
) -> None:
    bounds = (0.45, 0.12, 0.55, 0.25) if ppe_item == "HARD_HAT" else (0.45, 0.35, 0.55, 0.65)
    frame = overlapping_frame(detection(class_name, bounds))
    context = left_half_region() if outside_competitor else region()
    observations = PpePipeline(emit_missing=emit_missing).process(frame, context)

    assert all(item.ppe_item != ppe_item for item in observations)
    if outside_competitor:
        assert all(item.track_id == 1 for item in observations)


@pytest.mark.parametrize("reverse_people", [False, True])
@pytest.mark.parametrize("swap_track_ids", [False, True])
def test_ppe_pipeline_does_not_resolve_ambiguity_with_track_id_or_person_order(
    reverse_people: bool, swap_track_ids: bool
) -> None:
    frame = overlapping_frame(
        detection("NO-Hardhat", (0.45, 0.12, 0.55, 0.25)),
        reverse_people=reverse_people,
        swap_track_ids=swap_track_ids,
    )
    assert PpePipeline().process(frame, region()) == ()


def test_ppe_pipeline_preserves_independent_item_when_helmet_is_ambiguous() -> None:
    frame = overlapping_frame(
        detection("NO-Hardhat", (0.45, 0.12, 0.55, 0.25)),
        detection("Safety Vest", (0.15, 0.35, 0.30, 0.65)),
    )
    observations = PpePipeline().process(frame, region())
    assert [(item.track_id, item.ppe_item, item.status) for item in observations] == [
        (1, "SAFETY_VEST", "PRESENT")
    ]


def test_ppe_pipeline_ambiguous_evidence_cannot_be_overridden_by_unique_item_box() -> None:
    frame = overlapping_frame(
        detection("Hardhat", (0.15, 0.12, 0.25, 0.25)),
        detection("NO-Hardhat", (0.45, 0.12, 0.55, 0.25)),
    )
    assert PpePipeline().process(frame, region()) == ()


def test_ppe_pipeline_unique_item_keeps_its_owner_despite_overlap_elsewhere() -> None:
    frame = overlapping_frame(detection("NO-Hardhat", (0.15, 0.12, 0.25, 0.25)))
    observations = PpePipeline().process(frame, region())
    assert [(item.track_id, item.ppe_item, item.status) for item in observations] == [
        (1, "HARD_HAT", "MISSING")
    ]


def test_ambiguous_association_breaks_pending_temporal_confirmation() -> None:
    gate = TemporalPpeCandidateGate(confirmation_frames=3)
    pipeline = PpePipeline()
    clear_owner = overlapping_frame(detection("NO-Hardhat", (0.15, 0.12, 0.25, 0.25)))
    ambiguous_owner = overlapping_frame(detection("NO-Hardhat", (0.45, 0.12, 0.55, 0.25)))
    t0 = datetime(2026, 9, 30, 12, tzinfo=UTC)

    for index, frame in enumerate((clear_owner, clear_owner, ambiguous_owner, clear_owner)):
        candidates = gate.update(
            stream_id=frame.batch.stream_id,
            session_id=frame.batch.session_id,
            observed_at=t0 + timedelta(milliseconds=index * 100),
            active_track_ids=tuple(person.track_id for person in frame.persons),
            observations=pipeline.process(frame, region()),
        )
        assert candidates == ()


def test_empty_present_map_ignores_positive_while_negatives_work() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("Hardhat", (0.14, 0.12, 0.24, 0.25), 0.95),
            detection("NO-Safety Vest", (0.15, 0.35, 0.30, 0.70), 0.90),
        )
    )
    pipeline = PpePipeline(class_names_by_item={})
    observations = pipeline.process(tracked, region())
    assert [(item.ppe_item, item.status) for item in observations] == [
        ("SAFETY_VEST", "MISSING"),
    ]


def test_empty_negative_map_ignores_negative_while_positives_work() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("Hardhat", (0.14, 0.12, 0.24, 0.25), 0.95),
            detection("NO-Safety Vest", (0.15, 0.35, 0.30, 0.70), 0.90),
        )
    )
    pipeline = PpePipeline(missing_class_names_by_item={})
    observations = pipeline.process(tracked, region())
    assert [(item.ppe_item, item.status) for item in observations] == [
        ("HARD_HAT", "PRESENT"),
    ]


def test_both_empty_maps_emit_zero_observations_even_with_detections_and_emit_missing() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("Hardhat", (0.14, 0.12, 0.24, 0.25), 0.95),
            detection("NO-Safety Vest", (0.15, 0.35, 0.30, 0.70), 0.90),
        )
    )
    pipeline = PpePipeline(
        class_names_by_item={},
        missing_class_names_by_item={},
        emit_missing=True,
    )
    observations = pipeline.process(tracked, region())
    assert observations == ()


def test_none_mappings_preserve_default_behavior() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.10, 0.10, 0.35, 0.80), 0.9),
            detection("Hardhat", (0.14, 0.12, 0.24, 0.25), 0.95),
            detection("NO-Safety Vest", (0.15, 0.35, 0.30, 0.70), 0.90),
        )
    )
    pipeline = PpePipeline(class_names_by_item=None, missing_class_names_by_item=None)
    observations = pipeline.process(tracked, region())
    assert [(item.ppe_item, item.status) for item in observations] == [
        ("HARD_HAT", "PRESENT"),
        ("SAFETY_VEST", "MISSING"),
    ]


def test_custom_mapping_preserves_unsupported_item_and_collision_rejections() -> None:
    with pytest.raises(ValueError, match="unsupported PPE item"):
        PpePipeline(class_names_by_item={"UNSUPPORTED": ["hat"]})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="maps to multiple observations"):
        PpePipeline(
            class_names_by_item={"HARD_HAT": ["collision_hat"]},
            missing_class_names_by_item={"HARD_HAT": ["collision_hat"]},
        )


@pytest.mark.parametrize(
    ("negative_name", "item", "bounds"),
    [
        ("no-helmet", "HARD_HAT", (0.15, 0.12, 0.25, 0.25)),
        ("no-vest", "SAFETY_VEST", (0.15, 0.35, 0.30, 0.65)),
    ],
)
def test_native_negative_labels_reach_default_ppe_pipeline(
    negative_name: str, item: str, bounds: tuple[float, float, float, float]
) -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("person", (0.10, 0.10, 0.35, 0.80)).model_copy(update={"class_id": 3}),
            detection(negative_name, bounds).model_copy(
                update={"class_id": 1 if item == "HARD_HAT" else 2}
            ),
        )
    )
    observations = PpePipeline().process(tracked, region())
    assert [(obs.track_id, obs.ppe_item, obs.status) for obs in observations] == [
        (1, item, "MISSING")
    ]
    assert {(det.class_id, det.class_name) for det in tracked.batch.detections} == {
        (3, "person"),
        (1 if item == "HARD_HAT" else 2, negative_name),
    }


@pytest.mark.parametrize(
    ("positive", "negative", "bounds"),
    [
        ("helmet", "no-helmet", (0.15, 0.12, 0.25, 0.25)),
        ("vest", "no-vest", (0.15, 0.35, 0.30, 0.65)),
        ("Hardhat", "no-helmet", (0.15, 0.12, 0.25, 0.25)),
        ("helmet", "NO-Hardhat", (0.15, 0.12, 0.25, 0.25)),
        ("Safety Vest", "no-vest", (0.15, 0.35, 0.30, 0.65)),
        ("vest", "NO-Safety Vest", (0.15, 0.35, 0.30, 0.65)),
    ],
)
def test_native_contradictory_labels_remain_unknown(
    positive: str, negative: str, bounds: tuple[float, float, float, float]
) -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("person", (0.10, 0.10, 0.35, 0.80)),
            detection(positive, bounds),
            detection(negative, bounds),
        )
    )
    assert PpePipeline().process(tracked, region()) == ()


@pytest.mark.parametrize(
    ("negative", "item", "bounds"),
    [
        ("no-helmet", "HARD_HAT", (0.15, 0.12, 0.25, 0.25)),
        ("no-vest", "SAFETY_VEST", (0.15, 0.35, 0.30, 0.65)),
    ],
)
def test_native_duplicate_boxes_confirm_once_after_three_frames(
    negative: str, item: str, bounds: tuple[float, float, float, float]
) -> None:
    pipeline = PpePipeline()
    gate = TemporalPpeCandidateGate(confirmation_frames=3)
    tracker = IoUPersonTracker()
    t0 = datetime(2026, 10, 5, 12, tzinfo=UTC)
    candidates = []
    for index in range(4):
        current = batch(
            detection("person", (0.10, 0.10, 0.35, 0.80)),
            detection(negative, bounds),
            detection(negative, bounds, confidence=0.8),
        ).model_copy(update={"sequence_number": index + 1})
        tracked = tracker.update(current)
        emitted = gate.update(
            stream_id=current.stream_id,
            session_id=current.session_id,
            observed_at=t0 + timedelta(milliseconds=100 * index),
            active_track_ids=tuple(person.track_id for person in tracked.persons),
            observations=pipeline.process(tracked, region()),
        )
        assert len(emitted) == (1 if index == 2 else 0)
        candidates.extend(emitted)
    assert len(candidates) == 1
    assert candidates[0].ppe_item == item
    assert candidates[0].first_seen_at == t0


@pytest.mark.parametrize("unsupported", ["no-gloves", "no-boots", "no-harness", "NO-PPE"])
def test_unsupported_negative_labels_are_not_coerced_into_helmet_or_vest(
    unsupported: str,
) -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("person", (0.10, 0.10, 0.35, 0.80)),
            detection(unsupported, (0.15, 0.35, 0.30, 0.65)),
        )
    )
    assert PpePipeline().process(tracked, region()) == ()
