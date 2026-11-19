import pytest
from test_tracking import batch
from test_tracking_margin import person

from smartsite_ai.tracking import IoUPersonTracker


def test_weak_duplicate_never_starts_a_track_but_weak_real_observation_continues():
    tracker = IoUPersonTracker(new_track_confidence=0.5)
    first = tracker.update(batch(1, person(0.1, 0.5, 0.85), person(0.2, 0.6, 0.28)))
    assert len(first.persons) == 1
    segment = first.subject_segments[0].segment_id
    second = tracker.update(batch(2, person(0.11, 0.51, 0.32)))
    assert len(second.persons) == 1
    assert second.subject_segments[0].segment_id == segment
    assert second.persons[0].detection.confidence == 0.32
    third = tracker.update(batch(3, person(0.12, 0.52, 0.8), person(0.3, 0.7, 0.27)))
    assert len(third.persons) == 1
    assert third.subject_segments[0].segment_id == segment


def test_two_strong_people_are_not_capped_and_weak_only_startup_is_not_a_person_claim():
    tracker = IoUPersonTracker(new_track_confidence=0.5)
    first = tracker.update(batch(1, person(0.1, 0.3, 0.3)))
    assert not first.persons
    second = tracker.update(batch(2, person(0.1, 0.3, 0.9), person(0.6, 0.8, 0.85)))
    assert len(second.persons) == 2
    assert len(second.batch.detections) == 2


def test_empty_capture_still_ends_segment_even_when_weak_detection_later_matches():
    tracker = IoUPersonTracker(new_track_confidence=0.5)
    first = tracker.update(batch(1, person(0.1, 0.5, 0.8)))
    tracker.update(batch(2))
    later = tracker.update(batch(3, person(0.1, 0.5, 0.3)))
    assert later.subject_segments[0].segment_id != first.subject_segments[0].segment_id


def test_ambiguous_high_pass_cannot_revive_retired_owners_through_low_pass():
    tracker = IoUPersonTracker(new_track_confidence=0.5)
    first = tracker.update(batch(1, person(0.1, 0.5, 0.8), person(0.3, 0.7, 0.9)))
    later = tracker.update(batch(2, person(0.2, 0.6, 0.8), person(0.1, 0.5, 0.3)))
    assert len(later.retired_segments) == 2
    assert {s.segment_id for s in first.subject_segments}.isdisjoint(
        s.segment_id for s in later.subject_segments
    )


@pytest.mark.parametrize("high_right", [0.4, 0.22])
def test_low_box_competing_with_a_high_person_cannot_inherit_the_other_owner(high_right):
    tracker = IoUPersonTracker(new_track_confidence=0.5)
    first = tracker.update(batch(1, person(0, 0.4, 0.9), person(0.4, 0.8, 0.8)))
    second = tracker.update(batch(2, person(0, high_right, 0.9), person(0.15, 0.65, 0.3)))
    assert len(second.persons) == 1
    assert second.subject_segments[0].segment_id == first.subject_segments[0].segment_id
    assert first.subject_segments[1].segment_id in {r.segment_id for r in second.retired_segments}


@pytest.mark.parametrize("value", [True, 0, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_new_track_threshold_validation(value):
    with pytest.raises(ValueError, match="new_track_confidence"):
        IoUPersonTracker(new_track_confidence=value)
