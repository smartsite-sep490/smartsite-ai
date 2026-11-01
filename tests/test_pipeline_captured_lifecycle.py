"""Internal captured-empty frames are not safe scenes or empty legacy wire events."""

import pytest
from test_mf05_mf06_pipeline import batch, configuration, detection, make_pipeline

from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.tracking import IoUPersonTracker

EVENT = "00000000-0000-4000-8000-000000000019"


def person():
    return detection("person", (0.1, 0.1, 0.3, 0.8))


def test_captured_empty_frame_keeps_retirement_once_without_zone_exit_or_safe_claim():
    pipeline = make_pipeline()
    first = pipeline.observe_frame(
        batch(1, person()), region_configuration=configuration(), event_id=EVENT
    )
    empty = pipeline.observe_frame(batch(2), region_configuration=configuration(), event_id=EVENT)
    assert empty.observations == () and empty.subject_segments == ()
    assert empty.source_sequence_number == 2
    assert empty.captured_at == batch(2).captured_at.isoformat()
    assert [(s.segment_id, s.reason) for s in empty.retired_segments] == [
        (first.subject_segments[0].segment_id, "LOST_TRACK")
    ]
    repeated = pipeline.observe_frame(
        batch(3), region_configuration=configuration(), event_id=EVENT
    )
    assert repeated.observations == () and repeated.retired_segments == ()
    returned = pipeline.observe_frame(
        batch(4, person()), region_configuration=configuration(), event_id=EVENT
    )
    assert returned.subject_segments[0].segment_id != first.subject_segments[0].segment_id
    assert returned.subject_segments[0].previous_sequence_number is None


def test_legacy_observe_and_process_keep_none_and_call_tracker_once_per_frame():
    class CountingTracker(IoUPersonTracker):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def update(self, batch, *, view_revision=None):
            self.calls += 1
            return super().update(batch, view_revision=view_revision)

    tracker = CountingTracker()
    pipeline = Mf05Mf06Pipeline(
        tracker=tracker,
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=configuration().regions[0].region_id,
    )
    assert pipeline.observe(batch(1), region_configuration=configuration(), event_id=EVENT) is None
    assert pipeline.process(batch(2), region_configuration=configuration(), event_id=EVENT) is None
    assert tracker.calls == 2


def test_captured_empty_result_cannot_be_serialized_as_existing_event_versions():
    empty = make_pipeline().observe_frame(
        batch(1), region_configuration=configuration(), event_id=EVENT
    )
    with pytest.raises(ValueError):
        empty.to_event()
    with pytest.raises(ValueError):
        empty.to_partitioned_events(sequence_number=1)
    with pytest.raises(ValueError):
        empty.pack_with_tracking(sequence_number=1)


def test_nonempty_explicit_frame_matches_legacy_payload_and_hash():
    explicit = (
        make_pipeline()
        .observe_frame(batch(1, person()), region_configuration=configuration(), event_id=EVENT)
        .to_event()
    )
    legacy = make_pipeline().process(
        batch(1, person()), region_configuration=configuration(), event_id=EVENT
    )
    assert legacy is not None
    assert explicit.to_wire_dict() == legacy.to_wire_dict()
    assert explicit.compute_payload_hash() == legacy.compute_payload_hash()


def test_rejected_camera_does_not_create_a_lost_track_retirement():
    pipeline = make_pipeline()
    first = pipeline.observe_frame(
        batch(1, person()), region_configuration=configuration(), event_id=EVENT
    )
    with pytest.raises(ValueError):
        pipeline.observe_frame(
            batch(2).model_copy(update={"camera_external_id": "other"}),
            region_configuration=configuration(),
            event_id=EVENT,
        )
    next_frame = pipeline.observe_frame(
        batch(2, person()), region_configuration=configuration(), event_id=EVENT
    )
    assert next_frame.subject_segments[0].segment_id == first.subject_segments[0].segment_id
    assert next_frame.subject_segments[0].previous_sequence_number == 1
    assert next_frame.retired_segments == ()
