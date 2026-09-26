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


def test_zone_pipeline_emits_after_confirmed_outside_to_inside_transition() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()

    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    inside1 = tracker.update(batch(2, person((0.45, 0.20, 0.55, 0.60))))
    inside2 = tracker.update(batch(3, person((0.45, 0.20, 0.55, 0.60))))
    inside3 = tracker.update(batch(4, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(outside, configuration()) == ()
    assert zones.process(inside1, configuration()) == ()
    assert zones.process(inside2, configuration()) == ()
    entries = zones.process(inside3, configuration())

    assert len(entries) == 1
    assert entries[0].track_id == 1
    assert entries[0].region_id == REGION_ID
    assert entries[0].geometry_version == 7


def test_zone_pipeline_emits_initial_inside_occupancy_after_startup_confirmation() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    for sequence_number in (1, 2):
        inside = tracker.update(batch(sequence_number, person((0.45, 0.20, 0.55, 0.60))))
        assert zones.process(inside, configuration()) == ()
    inside = tracker.update(batch(3, person((0.45, 0.20, 0.55, 0.60))))
    entries = zones.process(inside, configuration())

    assert len(entries) == 1
    assert entries[0].track_id == 1


def test_zone_pipeline_resets_state_when_geometry_version_changes() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    inside = tracker.update(batch(2, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(outside, configuration()) == ()
    assert zones.process(inside, configuration()) == ()
    moved_back_outside = tracker.update(batch(3, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(moved_back_outside, configuration(geometry_version=8)) == ()


def test_zone_pipeline_uses_frame_hysteresis_at_polygon_boundary() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline(entry_confirmation_frames=2, exit_confirmation_frames=2)

    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(outside, configuration()) == ()
    inside = tracker.update(batch(2, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(inside, configuration()) == ()
    jitter_outside = tracker.update(batch(3, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(jitter_outside, configuration()) == ()
    inside_again = tracker.update(batch(4, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(inside_again, configuration()) == ()
    confirmed_inside = tracker.update(batch(5, person((0.45, 0.20, 0.55, 0.60))))
    assert len(zones.process(confirmed_inside, configuration())) == 1

    one_outside = tracker.update(batch(6, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(one_outside, configuration()) == ()
    returns_inside = tracker.update(batch(7, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(returns_inside, configuration()) == ()
    outside1 = tracker.update(batch(8, person((0.35, 0.20, 0.55, 0.39))))
    outside2 = tracker.update(batch(9, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(outside1, configuration()) == ()
    assert zones.process(outside2, configuration()) == ()
    reentry1 = tracker.update(batch(10, person((0.45, 0.20, 0.55, 0.60))))
    reentry2 = tracker.update(batch(11, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(reentry1, configuration()) == ()
    assert len(zones.process(reentry2, configuration())) == 1


def test_zone_pipeline_missing_frame_breaks_pending_entry_and_exit_streaks() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline(entry_confirmation_frames=2, exit_confirmation_frames=2)

    inside = tracker.update(batch(1, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(inside, configuration()) == ()
    zones.process(tracker.update(empty_batch(2)), configuration())
    inside_again = tracker.update(batch(3, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(inside_again, configuration()) == ()
    confirmed = tracker.update(batch(4, person((0.45, 0.20, 0.55, 0.60))))
    assert len(zones.process(confirmed, configuration())) == 1

    outside = tracker.update(batch(5, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(outside, configuration()) == ()
    zones.process(tracker.update(empty_batch(6)), configuration())
    outside_again = tracker.update(batch(7, person((0.35, 0.20, 0.55, 0.39))))
    assert zones.process(outside_again, configuration()) == ()
    assert zones.occupied_track_regions(
        stream_id="stream-01", session_id=SESSION, active_track_ids=(1,)
    ) == frozenset({(1, REGION_ID)})


def test_zone_pipeline_reports_occupancy_after_entry_without_duplicate_event() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline(entry_confirmation_frames=2)

    for sequence_number in (1, 2):
        tracked = tracker.update(batch(sequence_number, person((0.45, 0.20, 0.55, 0.60))))
        entries = zones.process(tracked, configuration())
    assert len(entries) == 1

    next_inside = tracker.update(batch(3, person((0.45, 0.20, 0.55, 0.60))))
    assert zones.process(next_inside, configuration()) == ()
    assert zones.occupied_track_regions(
        stream_id="stream-01", session_id=SESSION, active_track_ids=(1,)
    ) == frozenset({(1, REGION_ID)})


def test_zone_pipeline_reconfirms_after_geometry_or_session_reset() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline(entry_confirmation_frames=2)
    inside_detection = person((0.45, 0.20, 0.55, 0.60))

    zones.process(tracker.update(batch(1, inside_detection)), configuration())
    assert len(zones.process(tracker.update(batch(2, inside_detection)), configuration())) == 1

    assert (
        zones.process(tracker.update(batch(3, inside_detection)), configuration(geometry_version=8))
        == ()
    )
    assert (
        len(
            zones.process(
                tracker.update(batch(4, inside_detection)), configuration(geometry_version=8)
            )
        )
        == 1
    )

    new_session = UUID("00000000-0000-4000-8000-000000000099")
    first_new_session = batch(1, inside_detection).model_copy(update={"session_id": new_session})
    second_new_session = batch(2, inside_detection).model_copy(update={"session_id": new_session})
    assert zones.process(tracker.update(first_new_session), configuration(geometry_version=8)) == ()
    assert (
        len(zones.process(tracker.update(second_new_session), configuration(geometry_version=8)))
        == 1
    )


def empty_batch(sequence_number: int) -> DetectionBatch:
    return batch(sequence_number, person((0.35, 0.20, 0.55, 0.39))).model_copy(
        update={"detections": ()}
    )


def test_zone_pipeline_drops_track_state_after_the_tracker_grace_window() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    outside = tracker.update(batch(1, person((0.35, 0.20, 0.55, 0.39))))
    zones.process(outside, configuration())
    assert zones._inside

    for sequence_number in (2, 3):
        zones.process(tracker.update(empty_batch(sequence_number)), configuration())
    assert zones._inside

    zones.process(tracker.update(empty_batch(4)), configuration())
    assert zones._inside == {}


def test_zone_pipeline_drops_unconfirmed_initial_inside_streak_after_grace_window() -> None:
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline()
    initial_inside = tracker.update(batch(1, person((0.45, 0.20, 0.55, 0.60))))

    assert zones.process(initial_inside, configuration()) == ()
    assert zones._inside_streaks

    for sequence_number in (2, 3, 4):
        zones.process(tracker.update(empty_batch(sequence_number)), configuration())

    assert zones._inside_streaks == {}
    assert zones._outside_streaks == {}


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
