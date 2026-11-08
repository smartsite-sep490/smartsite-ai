"""Offline producer for the pinned C04 draft, outside the live transport."""

import re
from typing import Any, Literal
from uuid import UUID, uuid5

from smartsite_ai.domain.observations import ObservationFramePartition
from smartsite_ai.domain.tracked_frame_result import (
    AdmittedSubject,
    TrackedFrameReferences,
    TrackedFrameResult,
)
from smartsite_ai.pipelines.observation_frame import ObservationFrame

CapturedCoverageReason = Literal["INFERENCE_ERROR", "AMBIGUOUS_PROCESSING", "FRAME_DROPPED"]


def _source_sequence(frame: ObservationFrame, profile_sha256: str) -> int:
    sequence = frame.source_sequence_number
    if type(sequence) is not int or not 0 <= sequence <= 2**64 - 1:
        raise ValueError("Tracked frame requires an unsigned 64-bit source sequence")
    if not re.fullmatch("[0-9a-f]{64}", profile_sha256):
        raise ValueError("Tracked frame requires a canonical profile digest")
    return sequence


def _empty_part(frame: ObservationFrame, sequence: int) -> dict[str, Any]:
    partition = ObservationFramePartition(
        frameId=frame.event_id, sequenceNumber=str(sequence), partIndex=0, partCount=1
    )
    return {
        "cameraExternalId": frame.camera_external_id,
        "streamSessionId": frame.stream_session_id,
        "capturedAt": frame.captured_at,
        "frameDimensions": frame.frame_dimensions.model_dump(by_alias=True),
        "framePartition": partition.model_dump(by_alias=True),
        "observations": [],
        "evidence": [],
    }


def _tracked_part_id(event: dict[str, Any]) -> str:
    partition = event["framePartition"]
    return str(uuid5(UUID(partition["frameId"]), f"observation-v1.5-part:{partition['partIndex']}"))


def encode_incomplete_tracked_frame(
    frame: ObservationFrame,
    *,
    profile_sha256: str,
    reason: CapturedCoverageReason,
) -> tuple[dict[str, Any], ...]:
    """Keep observed data for review without constructing an expected admission seal."""
    if reason not in ("INFERENCE_ERROR", "AMBIGUOUS_PROCESSING", "FRAME_DROPPED"):
        raise ValueError("Invalid captured-frame coverage reason")
    sequence = _source_sequence(frame, profile_sha256)
    parts = (
        tuple(part.to_wire_dict() for part in frame.to_partitioned_events(sequence_number=sequence))
        if frame.observations
        else (_empty_part(frame, sequence),)
    )
    for part in parts:
        part.update(
            schemaVersion="1.5.0",
            eventId=_tracked_part_id(part),
            continuity={
                "schemaVersion": "1.0.0",
                "profileSha256": profile_sha256,
                "coverage": {"status": "INCOMPLETE", "reason": reason},
                "subjects": [],
                "retiredSegments": [],
            },
        )
    return parts


def encode_complete_tracked_frame(
    frame: ObservationFrame,
    *,
    scope_sha256: str,
    profile_sha256: str,
    predecessor: TrackedFrameResult | None = None,
) -> tuple[tuple[dict[str, Any], ...], TrackedFrameReferences]:
    """Encode a complete captured frame and the references expected from its seal."""
    sequence = _source_sequence(frame, profile_sha256)

    subjects = {}
    retirements = [
        {"segmentId": str(item.segment_id), "reason": item.reason}
        for item in frame.retired_segments
    ]
    if frame.observations or frame.subject_segments:
        packed = frame.pack_with_tracking(sequence_number=sequence)
        parts = [(part.event.to_wire_dict(), part.subjects) for part in packed]
    else:
        parts = [(_empty_part(frame, sequence), ())]
    if len(retirements) > len(parts) * 256:
        raise ValueError("Tracked retirements exceed the available part capacity")

    prior_subjects = (
        predecessor.subject_references()
        if isinstance(predecessor, TrackedFrameResult)
        and predecessor.scope_sha256 == scope_sha256
        and int(predecessor.sequence_number) + 1 == sequence
        else {}
    )
    for index, (event, refs) in enumerate(parts):
        event_id = _tracked_part_id(event)
        claims = []
        for ref in refs:
            subject = ref.subject
            segment_id = str(subject.segment_id)
            if subject.previous_sequence_number is None:
                link = {"kind": "START", "reason": subject.reason}
            else:
                prior = prior_subjects.get(segment_id)
                if prior is None or subject.previous_sequence_number != sequence - 1:
                    raise ValueError("Tracked continuation requires the exact admitted predecessor")
                previous_event_id, previous_person_index = prior
                if previous_event_id == event_id:
                    raise ValueError("Tracked continuation cannot reference its own event")
                link = {
                    "kind": "CONTINUE",
                    "previousEventId": previous_event_id,
                    "previousPersonObservationIndex": previous_person_index,
                }
            claims.append(
                {
                    "segmentId": segment_id,
                    "personObservationIndex": ref.person_observation_index,
                    "link": link,
                }
            )
            # The first part containing this segment supplies its original PERSON index.
            if segment_id not in subjects:
                subjects[segment_id] = AdmittedSubject(
                    segmentId=segment_id,
                    eventId=event_id,
                    personObservationIndex=ref.person_observation_index,
                )

        event.update(
            schemaVersion="1.5.0",
            eventId=event_id,
            continuity={
                "schemaVersion": "1.0.0",
                "profileSha256": profile_sha256,
                "coverage": {"status": "COMPLETE"},
                "subjects": claims,
                "retiredSegments": retirements[index * 256 : (index + 1) * 256],
            },
        )

    expected = TrackedFrameReferences(
        scopeSha256=scope_sha256,
        sequenceNumber=str(sequence),
        subjects=tuple(subjects.values()),
        retiredSegmentIds=tuple(item["segmentId"] for item in retirements),
    )
    return tuple(event for event, _ in parts), expected
