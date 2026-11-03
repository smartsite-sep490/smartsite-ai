"""Synthetic continuity boundaries, not physical identity qualification."""

from test_ppe_temporal import ppe_obs
from test_tracking import batch, detection

from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate
from smartsite_ai.tracking import IoUPersonTracker


def person(left: float, right: float):
    return detection("person", (left, 0.1, right, 0.8))


def ids(frame):
    return [item.track_id for item in frame.persons]


def test_merge_retires_both_old_tracks_without_dropping_person_or_bystander():
    tracker = IoUPersonTracker()
    assert ids(
        tracker.update(batch(1, person(0.1, 0.4), person(0.3, 0.6), person(0.75, 0.95)))
    ) == [1, 2, 3]
    merged = person(0.2, 0.5)
    source = batch(2, merged, person(0.75, 0.95))
    current = tracker.update(source)

    assert ids(current) == [3, 4]
    assert current.batch is source
    assert current.persons[1].detection is merged
    # A later separate detection must not resurrect retired IDs 1/2.
    later = tracker.update(batch(3, person(0.1, 0.4), person(0.3, 0.6), person(0.75, 0.95)))
    assert ids(later) == [3, 5, 6]


def test_split_allocates_two_fresh_ids_instead_of_arbitrary_old_owner():
    tracker = IoUPersonTracker()
    assert ids(tracker.update(batch(1, person(0.2, 0.5)))) == [1]

    split = tracker.update(batch(2, person(0.1, 0.4), person(0.3, 0.6)))

    assert ids(split) == [2, 3]
    assert [p.bounding_box.x1 for p in split.persons] == [0.1, 0.3]


def test_entire_ambiguous_chain_ends_but_disconnected_match_continues():
    tracker = IoUPersonTracker()
    assert ids(
        tracker.update(batch(1, person(0.05, 0.35), person(0.25, 0.55), person(0.75, 0.95)))
    ) == [1, 2, 3]
    # First detection fits both tracks; second fits only track 2. Neither
    # candidate may inherit track 1/2 through a greedy tie break.
    current = tracker.update(batch(2, person(0.15, 0.45), person(0.35, 0.65), person(0.75, 0.95)))

    assert ids(current) == [3, 4, 5]


def test_merged_person_cannot_inherit_previous_persons_missing_confirmation():
    tracker = IoUPersonTracker()
    gate = TemporalPpeCandidateGate(confirmation_frames=3)

    def observe(sequence, people):
        frame = tracker.update(batch(sequence, *people))
        active = ids(frame)
        return gate.update(
            stream_id=frame.batch.stream_id,
            session_id=frame.batch.session_id,
            observed_at=frame.batch.captured_at,
            active_track_ids=active,
            observations=tuple(ppe_obs(track_id) for track_id in active),
        )

    separated = (person(0.1, 0.4), person(0.3, 0.6))
    assert observe(1, separated) == ()
    assert observe(2, separated) == ()
    merged = (person(0.2, 0.5),)
    assert observe(3, merged) == ()
    assert observe(4, merged) == ()
    confirmed = observe(5, merged)
    assert len(confirmed) == 1
    assert confirmed[0].track_id == 3
    assert confirmed[0].first_seen_at == batch(3).captured_at
