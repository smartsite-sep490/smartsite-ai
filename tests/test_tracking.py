from datetime import UTC, datetime
from uuid import UUID

import pytest

from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson

SESSION = UUID("00000000-0000-4000-8000-000000000001")


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


@pytest.mark.parametrize("different_boxes", [False, True])
def test_tracked_frame_rejects_duplicate_track_id_regardless_of_person_box(
    different_boxes: bool,
) -> None:
    first = detection("person", (0.10, 0.10, 0.40, 0.80))
    second = detection("person", (0.60, 0.10, 0.90, 0.80)) if different_boxes else first
    detector_batch = batch(1, first, second)

    with pytest.raises(ValueError, match="track IDs must be unique within a frame"):
        TrackedFrame(
            batch=detector_batch,
            persons=(
                TrackedPerson(track_id=7, detection=first),
                TrackedPerson(track_id=7, detection=second),
            ),
        )

    # Rejection neither rewrites detector evidence nor selects a winning person.
    assert detector_batch.detections == (first, second)


def test_tracked_frame_accepts_unique_tracks_even_when_person_boxes_are_identical() -> None:
    observed = detection("person", (0.10, 0.10, 0.40, 0.80))
    persons = (
        TrackedPerson(track_id=0, detection=observed),
        TrackedPerson(track_id=1, detection=observed),
    )
    detector_batch = batch(1, observed, observed)

    frame = TrackedFrame(batch=detector_batch, persons=persons)

    assert frame.batch is detector_batch
    assert frame.persons == persons
    # Unique track IDs do not prove these are different physical Workers.


def test_tracked_frame_accepts_no_current_person_tracks() -> None:
    detector_batch = batch(1)

    frame = TrackedFrame(batch=detector_batch, persons=())

    assert frame.batch is detector_batch
    assert frame.persons == ()


def test_iou_tracker_keeps_person_id_and_ignores_ppe_detections() -> None:
    tracker = IoUPersonTracker(iou_threshold=0.3)

    first = tracker.update(
        batch(
            1,
            detection("person", (0.10, 0.10, 0.40, 0.80)),
            detection("helmet", (0.18, 0.12, 0.25, 0.20)),
        )
    )
    second = tracker.update(batch(2, detection("person", (0.12, 0.10, 0.42, 0.80))))

    assert [person.track_id for person in first.persons] == [1]
    assert [person.track_id for person in second.persons] == [1]


def test_iou_tracker_survives_configured_missed_frames_then_allocates_new_id() -> None:
    tracker = IoUPersonTracker(max_missed_frames=1)
    tracker.update(batch(1, detection("person", (0.10, 0.10, 0.40, 0.80))))

    assert tracker.update(batch(2)).persons == ()
    assert [
        person.track_id
        for person in tracker.update(
            batch(3, detection("person", (0.10, 0.10, 0.40, 0.80)))
        ).persons
    ] == [1]
    assert tracker.update(batch(4)).persons == ()
    assert tracker.update(batch(5)).persons == ()
    assert [
        person.track_id
        for person in tracker.update(
            batch(6, detection("person", (0.10, 0.10, 0.40, 0.80)))
        ).persons
    ] == [2]


def test_iou_tracker_resets_ids_for_new_session_and_rejects_out_of_order_frames() -> None:
    tracker = IoUPersonTracker()
    tracker.update(batch(3, detection("person", (0.10, 0.10, 0.40, 0.80))))

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update(batch(3, detection("person", (0.10, 0.10, 0.40, 0.80))))

    next_session = batch(1, detection("person", (0.10, 0.10, 0.40, 0.80))).model_copy(
        update={"session_id": UUID("00000000-0000-4000-8000-000000000002")}
    )
    assert [person.track_id for person in tracker.update(next_session).persons] == [1]


def test_tracker_rejects_camera_change_within_session_without_mutating_tracks() -> None:
    tracker = IoUPersonTracker()
    person = detection("person", (0.10, 0.10, 0.40, 0.80))
    tracker.update(batch(1, person))
    other_camera = batch(2, person).model_copy(update={"camera_external_id": "camera-02"})

    with pytest.raises(ValueError, match="camera.*session"):
        tracker.update(other_camera)

    # A rejected batch must not consume sequence 2 or change the original source.
    assert [item.track_id for item in tracker.update(batch(2, person)).persons] == [1]


def test_tracker_accepts_new_camera_only_with_new_session_and_clears_old_tracks() -> None:
    tracker = IoUPersonTracker()
    tracker.update(
        batch(
            1,
            detection("person", (0.10, 0.10, 0.40, 0.80)),
            detection("person", (0.60, 0.10, 0.90, 0.80)),
        )
    )
    other_source = batch(1, detection("person", (0.60, 0.10, 0.90, 0.80))).model_copy(
        update={
            "camera_external_id": "camera-02",
            "session_id": UUID("00000000-0000-4000-8000-000000000002"),
        }
    )

    assert [person.track_id for person in tracker.update(other_source).persons] == [1]
