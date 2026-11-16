from dataclasses import replace
from uuid import UUID

import pytest
from test_tracked_transport import packet

from smartsite_ai.domain.tracked_frame_result import TrackedFrameResult
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent
from smartsite_ai.pipelines.observation_frame import ObservationFrame
from smartsite_ai.pipelines.tracked_frame_encoder import (
    encode_complete_tracked_frame,
    encode_incomplete_tracked_frame,
)
from smartsite_ai.tracking.models import RetiredSegment, SubjectSegment


def frame():
    original, *_ = packet()
    event = TrackedObservationEvent.model_validate(original)
    person = event.observations[0].model_copy(update={"track_id": 1})
    ppe = event.observations[1].model_copy(update={"track_id": 1})
    return ObservationFrame(
        event_id=event.frame_partition.frame_id,
        schema_version="1.1.0",
        camera_external_id=event.camera_external_id,
        stream_session_id=event.stream_session_id,
        captured_at=event.captured_at,
        frame_dimensions=event.frame_dimensions,
        observations=(person, ppe),
        source_sequence_number=0,
        subject_segments=(
            SubjectSegment(1, UUID(event.continuity.subjects[0].segment_id), None, "INITIAL"),
        ),
        requires_tracked_wire=True,
    )


def encode(value, predecessor=None):
    return encode_complete_tracked_frame(
        value, scope_sha256="a" * 64, profile_sha256="b" * 64, predecessor=predecessor
    )


def test_encoder_retains_original_indices_and_requires_admitted_previous_frame():
    value = frame()
    parts, refs = encode(value)
    assert len(parts) == 1
    assert parts[0]["schemaVersion"] == "1.7.0"
    assert parts[0]["observations"] == [item.to_wire_dict() for item in value.observations]
    assert refs.subjects[0].person_index == 0
    claim = parts[0]["continuity"]["subjects"][0]
    assert claim["link"] == {"kind": "START", "reason": "INITIAL"}
    second = replace(
        value,
        event_id="00000000-0000-4000-8000-000000000055",
        source_sequence_number=1,
        subject_segments=(
            replace(value.subject_segments[0], previous_sequence_number=0, reason=None),
        ),
    )
    with pytest.raises(ValueError, match="predecessor"):
        encode(second)
    seal = TrackedFrameResult(kind="ADMITTED_TRACKED_FRAME", **refs.model_dump(by_alias=True))
    next_parts, _ = encode(second, seal)
    assert next_parts[0]["continuity"]["subjects"][0]["link"] == {
        "kind": "CONTINUE",
        "previousEventId": parts[0]["eventId"],
        "previousPersonObservationIndex": 0,
    }
    for event in (*parts, *next_parts):
        TrackedObservationEvent.model_validate(event)


def test_large_people_frame_packs_without_losing_or_reassigning_subjects():
    value = frame()
    people = tuple(value.observations[0].model_copy(update={"track_id": i}) for i in range(1, 261))
    subjects = tuple(SubjectSegment(i, UUID(int=i + 1000), None, "INITIAL") for i in range(1, 261))
    parts, refs = encode(replace(value, observations=people, subject_segments=subjects))
    assert len(parts) == 2 and len(refs.subjects) == 260
    assert sum(len(item["observations"]) for item in parts) == 260
    for item in parts:
        TrackedObservationEvent.model_validate(item)
    for subject in refs.subjects:
        original = next(item for item in parts if item["eventId"] == subject.event_id)
        assert (
            original["observations"][subject.person_index]["trackId"]
            == UUID(subject.segment_id).int - 1000
        )


def test_empty_captured_frame_transmits_retirement_without_inventing_safe_or_person():
    value = frame()
    empty = replace(
        value,
        observations=(),
        subject_segments=(),
        retired_segments=(RetiredSegment(value.subject_segments[0].segment_id, "LOST_TRACK"),),
    )
    parts, refs = encode(empty)
    assert parts[0]["observations"] == [] and refs.subjects == ()
    assert refs.retired_segment_ids == (str(value.subject_segments[0].segment_id),)
    TrackedObservationEvent.model_validate(parts[0])
    incomplete = encode_incomplete_tracked_frame(
        empty, profile_sha256="b" * 64, reason="INFERENCE_ERROR"
    )
    assert incomplete[0]["continuity"]["subjects"] == []
    assert incomplete[0]["continuity"]["retiredSegments"] == []
    TrackedObservationEvent.model_validate(incomplete[0])


def test_camera_frame_cannot_leak_into_legacy_serialization():
    value = frame()
    for method in (value.to_event, value.to_events):
        with pytest.raises(ValueError, match="tracked encoder"):
            method()
