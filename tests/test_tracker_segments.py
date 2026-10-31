"""Internal lifecycle claims, not qualified physical identity or wire admission."""

from datetime import timedelta
from uuid import uuid4

import pytest
from test_tracking import batch, detection

from smartsite_ai.tracking import IoUPersonTracker


def person(left=0.1, right=0.4):
    return detection("person", (left, 0.1, right, 0.8))


def claim(frame):
    assert len(frame.subject_segments) == 1
    return frame.subject_segments[0]


def test_adjacent_unique_match_preserves_segment_with_original_previous_sequence():
    tracker = IoUPersonTracker()
    first = claim(tracker.update(batch(1, person())))
    second = claim(tracker.update(batch(2, person())))
    assert first.segment_id == second.segment_id
    assert first.reason == "INITIAL" and first.previous_sequence_number is None
    assert second.reason is None and second.previous_sequence_number == 1


def test_empty_frame_retires_segment_once_even_when_numeric_track_can_reappear():
    tracker = IoUPersonTracker()
    first_frame = tracker.update(batch(1, person()))
    first = claim(first_frame)
    empty = tracker.update(batch(2))
    assert empty.persons == () and empty.subject_segments == ()
    assert [(item.segment_id, item.reason) for item in empty.retired_segments] == [
        (first.segment_id, "LOST_TRACK")
    ]
    later = tracker.update(batch(3, person()))
    assert later.persons[0].track_id == first_frame.persons[0].track_id
    assert claim(later).segment_id != first.segment_id
    assert later.retired_segments == ()


def test_merge_retires_ambiguous_segments_without_breaking_disconnected_bystander():
    tracker = IoUPersonTracker()
    first = tracker.update(batch(1, person(0.1, 0.4), person(0.3, 0.6), person(0.75, 0.95)))
    second = tracker.update(batch(2, person(0.2, 0.5), person(0.75, 0.95)))
    assert {item.segment_id for item in second.retired_segments} == {
        item.segment_id for item in first.subject_segments[:2]
    }
    assert all(item.reason == "AMBIGUOUS_ASSOCIATION" for item in second.retired_segments)
    assert second.subject_segments[0].segment_id == first.subject_segments[2].segment_id
    assert second.subject_segments[1].reason == "INITIAL"


def test_view_change_retires_old_segment_and_same_view_refresh_keeps_new_one():
    tracker = IoUPersonTracker()
    first = claim(tracker.update(batch(1, person()), view_revision=1))
    changed = tracker.update(batch(2, person()), view_revision=2)
    assert claim(changed).segment_id != first.segment_id
    assert claim(changed).reason == "VIEW_CHANGE"
    assert changed.retired_segments[0].segment_id == first.segment_id
    assert changed.retired_segments[0].reason == "VIEW_CHANGE"
    assert (
        claim(tracker.update(batch(3, person()), view_revision=2)).segment_id
        == claim(changed).segment_id
    )


def test_sequence_gap_capture_regression_and_model_change_never_continue_segment():
    for changed, reason in (
        (batch(3, person()), "FRAME_DISCONTINUITY"),
        (
            batch(2, person()).model_copy(
                update={"captured_at": batch(1).captured_at - timedelta(seconds=1)}
            ),
            "FRAME_DISCONTINUITY",
        ),
        (batch(2, person()).model_copy(update={"model_sha256": "b" * 64}), "PROFILE_CHANGE"),
        (batch(2, person()).model_copy(update={"frame_width": 800}), "VIEW_CHANGE"),
    ):
        tracker = IoUPersonTracker()
        first = claim(tracker.update(batch(1, person())))
        result = tracker.update(changed)
        assert claim(result).segment_id != first.segment_id
        assert claim(result).reason == reason
        assert result.retired_segments[0].reason == reason


def test_new_session_never_reuses_segment_or_emits_old_scope_retirement_as_current():
    tracker = IoUPersonTracker()
    old = claim(tracker.update(batch(1, person())))
    changed = tracker.update(batch(1, person()).model_copy(update={"session_id": uuid4()}))
    assert claim(changed).segment_id != old.segment_id
    assert claim(changed).reason == "RECONNECT"
    assert changed.retired_segments == ()


def test_rejected_batch_does_not_advance_or_replace_segment_predecessor():
    tracker = IoUPersonTracker()
    first = claim(tracker.update(batch(1, person()), view_revision=1))
    for wrong, revision in (
        (batch(1, person()), 2),
        (batch(2, person()).model_copy(update={"camera_external_id": "other-camera"}), 2),
        (batch(2, person()), True),
    ):
        with pytest.raises(ValueError):
            tracker.update(wrong, view_revision=revision)
    second = claim(tracker.update(batch(2, person()), view_revision=1))
    assert second.segment_id == first.segment_id
    assert second.previous_sequence_number == 1


def test_retirement_is_not_reemitted_or_resurrected_after_multiple_empty_frames():
    tracker = IoUPersonTracker()
    first = claim(tracker.update(batch(1, person())))
    assert tracker.update(batch(2)).retired_segments[0].segment_id == first.segment_id
    for sequence in (3, 4):
        empty = tracker.update(batch(sequence))
        assert empty.retired_segments == () and empty.subject_segments == ()
    returned = claim(tracker.update(batch(5, person())))
    assert returned.segment_id != first.segment_id
