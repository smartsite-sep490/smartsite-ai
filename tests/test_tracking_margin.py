"""Experimental geometric association, never a physical Worker identity claim."""

import pytest
from test_tracking import batch, detection

from smartsite_ai.tracking import IoUPersonTracker


def person(left, right, confidence=0.9):
    return detection("person", (left, 0.1, right, 0.8), confidence)


def segments(frame):
    return {
        p.bounding_box.x1: s.segment_id
        for p, s in zip(frame.persons, frame.subject_segments, strict=True)
    }


def test_explicit_margin_preserves_distinguishable_overlaps_despite_confidence_reordering():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5, 0.95), person(0.3, 0.7, 0.8)))
    second = tracker.update(batch(2, person(0.1, 0.5, 0.75), person(0.3, 0.7, 0.99)))
    assert segments(second) == segments(first)
    assert second.retired_segments == ()


@pytest.mark.parametrize("margin", [None, 0.8])
def test_default_or_insufficient_margin_keeps_conservative_retirement(margin):
    tracker = IoUPersonTracker(association_iou_margin=margin)
    first = tracker.update(batch(1, person(0.1, 0.5), person(0.3, 0.7)))
    second = tracker.update(batch(2, person(0.1, 0.5), person(0.3, 0.7)))
    assert set(segments(first).values()).isdisjoint(segments(second).values())
    assert len(second.retired_segments) == 2


@pytest.mark.parametrize(
    "next_people",
    [
        (person(0.2, 0.6),),
        (person(0.2, 0.6), person(0.2, 0.6)),
        (person(0.1, 0.5), person(0.3, 0.7), person(0.2, 0.6)),
    ],
)
def test_merge_tie_or_extra_competitor_never_inherits_prior_segments(next_people):
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5), person(0.3, 0.7)))
    second = tracker.update(batch(2, *next_people))
    assert set(segments(first).values()).isdisjoint(s.segment_id for s in second.subject_segments)
    assert len(second.retired_segments) == 2


def test_empty_frame_cannot_bridge_segment_under_margin_mode():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5)))
    tracker.update(batch(2))
    later = tracker.update(batch(3, person(0.1, 0.5)))
    assert first.subject_segments[0].segment_id != later.subject_segments[0].segment_id


def test_weak_column_retires_entire_balanced_component_including_strong_pair():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5), person(0.3, 0.7)))
    second = tracker.update(batch(2, person(0.1, 0.5), person(0.22, 0.62)))
    assert set(segments(first).values()).isdisjoint(segments(second).values())
    assert len(second.retired_segments) == 2


def test_split_and_unaffected_bystander_keep_distinct_lifecycle_boundaries():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.2, 0.5), person(0.8, 0.95)))
    second = tracker.update(batch(2, person(0.1, 0.4), person(0.3, 0.6), person(0.8, 0.95)))
    old = segments(first)
    current = segments(second)
    assert current[0.8] == old[0.8]
    assert old[0.2] not in current.values()
    assert len(second.retired_segments) == 1


def test_crossing_through_tied_boxes_cannot_resurrect_old_segments():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5), person(0.3, 0.7)))
    middle = tracker.update(batch(2, person(0.2, 0.6), person(0.2, 0.6)))
    later = tracker.update(batch(3, person(0.1, 0.5), person(0.3, 0.7)))
    assert set(segments(first).values()).isdisjoint(s.segment_id for s in middle.subject_segments)
    assert set(segments(first).values()).isdisjoint(s.segment_id for s in later.subject_segments)


def test_explicit_margin_does_not_bridge_a_skipped_source_frame():
    tracker = IoUPersonTracker(association_iou_margin=0.2)
    first = tracker.update(batch(1, person(0.1, 0.5), person(0.3, 0.7)))
    next_frame = tracker.update(batch(3, person(0.1, 0.5), person(0.3, 0.7)))
    assert set(segments(first).values()).isdisjoint(
        s.segment_id for s in next_frame.subject_segments
    )
    assert {r.reason for r in next_frame.retired_segments} == {"FRAME_DISCONTINUITY"}


@pytest.mark.parametrize("margin", [True, False, 0, -0.1, 1.1, float("nan"), float("inf"), "0.2"])
def test_margin_rejects_invalid_values(margin):
    with pytest.raises(ValueError, match="association_iou_margin"):
        IoUPersonTracker(association_iou_margin=margin)
