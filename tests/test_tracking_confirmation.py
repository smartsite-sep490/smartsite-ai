"""New Person admission delays evidence without claiming identity across gaps."""

from datetime import timedelta
from uuid import uuid4

import pytest
from test_ppe_pipeline import region
from test_tracking import batch, detection
from test_tracking_margin import person

from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson


def tracker(**options):
    return IoUPersonTracker(new_track_confirmation_frames=3, new_track_confidence=0.5, **options)


def test_pending_tracks_keep_raw_observations_and_start_only_at_confirmation():
    subject = person(0.1, 0.3)
    pending = tracker()
    for sequence in (1, 2):
        raw = batch(sequence, subject)
        current = pending.update(raw)
        assert current.batch is raw
        assert not current.persons and not current.subject_segments
        assert current.provisional_persons[0].detection == subject
    confirmed = pending.update(batch(3, subject))
    assert len(confirmed.persons) == len(confirmed.subject_segments) == 1
    assert confirmed.subject_segments[0].previous_sequence_number is None
    assert not confirmed.provisional_persons
    next_frame = pending.update(batch(4, person(0.11, 0.31, 0.3)))
    assert next_frame.subject_segments[0].segment_id == confirmed.subject_segments[0].segment_id


def test_default_still_admits_immediately_and_short_high_confidence_track_does_not():
    subject = person(0.1, 0.3)
    assert len(IoUPersonTracker().update(batch(1, subject)).persons) == 1
    pending = tracker()
    assert not pending.update(batch(1, subject)).persons
    assert not pending.update(batch(2)).persons
    assert not pending.update(batch(3, subject)).persons


def test_weak_observation_does_not_complete_new_track_confirmation():
    pending = tracker()
    for sequence, confidence in enumerate((0.9, 0.3, 0.9, 0.9), 1):
        assert not pending.update(batch(sequence, person(0.1, 0.3, confidence))).persons
    assert len(pending.update(batch(5, person(0.1, 0.3))).persons) == 1


@pytest.mark.parametrize(
    "change", ["sequence", "time", "gap", "model", "dimensions", "view", "session"]
)
def test_discontinuity_cannot_complete_a_pending_track(change):
    pending = tracker(max_segment_gap=timedelta(seconds=3))
    pending.update(batch(1, person(0.1, 0.3)), view_revision=1)
    pending.update(batch(2, person(0.1, 0.3)), view_revision=1)
    current = batch(3, person(0.1, 0.3))
    updates = {
        "sequence": {"sequence_number": 4},
        "time": {"captured_at": batch(2).captured_at},
        "gap": {"captured_at": batch(2).captured_at + timedelta(seconds=4)},
        "model": {"model_sha256": "b" * 64},
        "dimensions": {"frame_width": 1280},
        "session": {"session_id": uuid4()},
    }
    current = current.model_copy(update=updates.get(change, {}))
    assert not pending.update(current, view_revision=2 if change == "view" else 1).persons


def test_loss_retires_once_and_reappearance_requires_fresh_confirmation():
    pending = tracker(max_missed_frames=2)
    for sequence in (1, 2, 3):
        confirmed = pending.update(batch(sequence, person(0.1, 0.3)))
    original = confirmed.subject_segments[0].segment_id
    missing = pending.update(batch(4))
    assert [s.segment_id for s in missing.retired_segments] == [original]
    for sequence in (5, 6):
        current = pending.update(batch(sequence, person(0.1, 0.3)))
        assert not current.persons and not current.retired_segments
    returned = pending.update(batch(7, person(0.1, 0.3)))
    assert returned.subject_segments[0].segment_id != original
    assert returned.subject_segments[0].previous_sequence_number is None


def test_two_people_confirm_independently_and_ambiguous_merge_does_not_inherit():
    pending = tracker()
    for sequence in (1, 2, 3):
        confirmed = pending.update(batch(sequence, person(0.1, 0.4), person(0.6, 0.9)))
    assert len(confirmed.persons) == 2
    assert len({s.segment_id for s in confirmed.subject_segments}) == 2
    merged = pending.update(batch(4, person(0.1, 0.9)))
    assert not merged.persons
    assert len(merged.retired_segments) == 2


def test_pending_person_still_blocks_ambiguous_ppe_for_confirmed_neighbor():
    pending = tracker(association_iou_margin=0.1)
    primary = person(0.1, 0.5)
    for sequence in (1, 2, 3):
        pending.update(batch(sequence, primary))
    shared_helmet = detection("helmet", (0.44, 0.15, 0.49, 0.25))
    mixed = pending.update(batch(4, primary, person(0.4, 0.8), shared_helmet))
    assert len(mixed.persons) == len(mixed.provisional_persons) == 1
    assert PpePipeline().process(mixed, region()) == ()


def test_provisional_and_confirmed_people_cannot_share_one_track_id():
    subject = TrackedPerson(1, person(0.1, 0.3))
    with pytest.raises(ValueError, match="unique"):
        TrackedFrame(batch(1), (subject,), provisional_persons=(subject,))


def test_pending_track_produces_only_empty_v17_frame_without_ppe_or_zone_backfill():
    from test_mf05_mf06_bundle_policy import PPE_REGION_ID, scene_batch
    from test_tracked_policy_pipeline import policy_bundle

    from smartsite_ai.pipelines import Mf05Mf06Pipeline, RestrictedZonePipeline
    from smartsite_ai.pipelines.tracked_frame_encoder import encode_complete_tracked_frame

    pipeline = Mf05Mf06Pipeline(
        tracker=tracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(entry_confirmation_frames=1),
        ppe_region_id=PPE_REGION_ID,
        tracked_v17=True,
    )
    for sequence in (1, 2):
        frame = pipeline.observe_frame(
            scene_batch(sequence), region_configuration=policy_bundle(), event_id=str(uuid4())
        )
        assert not frame.observations and not frame.subject_segments
        parts, _ = encode_complete_tracked_frame(
            frame, scope_sha256="a" * 64, profile_sha256="b" * 64
        )
        assert parts[0]["observations"] == []
    confirmed = pipeline.observe_frame(
        scene_batch(3), region_configuration=policy_bundle(), event_id=str(uuid4())
    )
    assert {item.type for item in confirmed.observations} == {"PERSON", "PPE", "ZONE_ENTRY"}
    assert len(confirmed.subject_segments) == 1
    assert confirmed.subject_segments[0].previous_sequence_number is None


def test_reconfirmed_subject_cannot_inherit_prior_zone_occupancy():
    from test_mf05_mf06_bundle_policy import PPE_REGION_ID, scene_batch
    from test_tracked_policy_pipeline import policy_bundle

    from smartsite_ai.pipelines import Mf05Mf06Pipeline, RestrictedZonePipeline

    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(new_track_confirmation_frames=2, max_missed_frames=2),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(entry_confirmation_frames=1),
        ppe_region_id=PPE_REGION_ID,
        tracked_v17=True,
    )
    observed = []
    for sequence in (1, 2, 3, 4, 5):
        raw = scene_batch(sequence)
        if sequence == 3:
            raw = raw.model_copy(update={"detections": ()})
        observed.append(
            pipeline.observe_frame(raw, region_configuration=policy_bundle(), event_id=str(uuid4()))
        )
    assert sum(o.type == "ZONE_ENTRY" for o in observed[1].observations) == 1
    assert sum(o.type == "ZONE_ENTRY" for o in observed[4].observations) == 1
    first_person = next(o for o in observed[1].observations if o.type == "PERSON")
    returned_person = next(o for o in observed[4].observations if o.type == "PERSON")
    assert first_person.track_id != returned_person.track_id


@pytest.mark.parametrize("invalid", [0, 65, True, 1.5, None])
def test_confirmation_bound_is_a_strict_positive_integer(invalid):
    with pytest.raises(ValueError, match="new_track_confirmation_frames"):
        IoUPersonTracker(new_track_confirmation_frames=invalid)
