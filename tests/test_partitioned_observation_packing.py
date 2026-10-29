import hashlib
from dataclasses import replace
from datetime import datetime
from uuid import UUID

import pytest
from test_observation_capacity import crowd_frame, observation_counts

from smartsite_ai.domain.observations import (
    PpeObservation,
    TechnicalObservationEvent,
    ZoneEntryObservation,
)
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import (
    OutboxConflict,
    OutboxEnqueueOutcome,
    SqliteEventOutbox,
)
from smartsite_ai.pipelines.ppe_temporal import events_for_delivery


def test_partition_packing_preserves_crowd_context_and_deterministic_versioned_ids():
    frame = crowd_frame()
    parts = frame.to_partitioned_events(sequence_number=18_446_744_073_709_551_615)
    assert [len(p.observations) for p in parts] == [252, 28]
    assert observation_counts(o for p in parts for o in p.observations) == observation_counts(
        frame.observations
    )
    assert [p.to_wire_dict() for p in parts] == [
        p.to_wire_dict()
        for p in frame.to_partitioned_events(sequence_number=18_446_744_073_709_551_615)
    ]
    assert not {p.event_id for p in parts} & {p.event_id for p in frame.to_events()}
    for index, part in enumerate(parts):
        assert part.schema_version == "1.4.0"
        assert part.frame_partition.to_wire_dict() == {
            "frameId": frame.event_id,
            "sequenceNumber": "18446744073709551615",
            "partIndex": index,
            "partCount": 2,
        }
        assert TechnicalObservationEvent.model_validate(part.to_wire_dict()) == part
        people = [o.track_id for o in part.observations if o.type == "PERSON"]
        assert all(people.count(o.track_id) == 1 for o in part.observations)


def test_single_partition_has_metadata_while_legacy_keeps_original_id_and_shape():
    frame = crowd_frame()
    frame = replace(frame, observations=frame.observations[:28])
    (legacy,) = frame.to_events()
    (part,) = frame.to_partitioned_events(sequence_number=0)
    assert legacy.event_id == frame.event_id
    assert "framePartition" not in legacy.to_wire_dict()
    assert part.event_id != legacy.event_id
    assert part.frame_partition.part_count == 1
    assert part.frame_partition.sequence_number == "0"


def test_oversized_subject_repeats_only_person_context_in_partitioned_wire():
    frame = crowd_frame()
    person = frame.observations[0]
    zones = tuple(
        ZoneEntryObservation(
            type="ZONE_ENTRY",
            trackId=person.track_id,
            regionId=str(UUID(int=i + 200, version=4)),
            geometryVersion=1,
        )
        for i in range(300)
    )
    frame = replace(frame, observations=(person, *zones))
    parts = frame.to_partitioned_events(sequence_number=3)
    assert [len(p.observations) for p in parts] == [256, 46]
    assert all(p.observations[0] == person for p in parts)
    assert observation_counts(
        o for p in parts for o in p.observations if o.type != "PERSON"
    ) == observation_counts(zones)


def test_small_partition_rejects_orphan_subject_without_fabricating_person():
    frame = crowd_frame()
    frame = replace(
        frame, observations=(next(o for o in frame.observations if o.type == "ZONE_ENTRY"),)
    )
    with pytest.raises(ValueError, match="PERSON"):
        frame.to_partitioned_events(sequence_number=3)


@pytest.mark.parametrize("sequence", [-1, 18_446_744_073_709_551_616, True, 1.0, "1"])
def test_partition_packing_rejects_invalid_source_sequence(sequence):
    with pytest.raises(ValueError):
        crowd_frame().to_partitioned_events(sequence_number=sequence)


def test_delivery_filters_entire_frame_before_setting_partition_count():
    frame = crowd_frame()
    frame = replace(
        frame,
        observations=tuple(
            PpeObservation.model_validate({**o.to_wire_dict(), "status": "MISSING"})
            if o.type == "PPE"
            else o
            for o in frame.observations
        ),
    )
    (part,) = events_for_delivery(frame, partition_sequence_number=3)
    assert len(part.observations) == 100
    assert part.frame_partition.part_count == 1
    assert part.schema_version == "1.4.0"
    assert all(o.type != "PPE" for o in part.observations)


def test_partition_delivery_does_not_reinterpret_an_already_bounded_event():
    frame = crowd_frame()
    part = frame.to_events()[0]
    with pytest.raises(ValueError, match="full ObservationFrame"):
        events_for_delivery(part, partition_sequence_number=3)


def test_legacy_delivery_refuses_uncorrelated_multipart_instead_of_inflating_counts():
    with pytest.raises(ValueError, match="v1.4"):
        events_for_delivery(crowd_frame())


@pytest.mark.anyio
async def test_generated_partitions_bind_real_jpeg_and_replay_atomically(tmp_path):
    frame = crowd_frame()
    pixels = FrameEnvelope(
        stream_id="packing-test",
        session_id=UUID(frame.stream_session_id),
        camera_external_id=frame.camera_external_id,
        captured_at=datetime.fromisoformat(frame.captured_at),
        width=frame.frame_dimensions.width,
        height=frame.frame_dimensions.height,
        sequence_number=3,
        payload=bytes(frame.frame_dimensions.width * frame.frame_dimensions.height * 3),
    )
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    parts = tuple(
        [
            await publisher.publish(frame=pixels, event=part)
            for part in events_for_delivery(frame, partition_sequence_number=3)
        ]
    )
    assert len(parts) == 2
    jpeg_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.jpg")}
    assert len(jpeg_hashes) == 1
    assert {p.evidence[0].sha256 for p in parts} == jpeg_hashes
    path = tmp_path / "parts.sqlite3"
    outbox = SqliteEventOutbox(path)
    assert await outbox.enqueue_many(parts) == (OutboxEnqueueOutcome.ENQUEUED,) * 2
    assert await outbox.enqueue_many(parts) == (OutboxEnqueueOutcome.IDEMPOTENT,) * 2
    replay = await SqliteEventOutbox(path).ready()
    assert {p.event.event_id: p.event.to_wire_dict() for p in replay} == {
        p.event_id: p.to_wire_dict() for p in parts
    }
    assert all(p.payload_hash == p.event.compute_payload_hash() for p in replay)

    conflict_box = SqliteEventOutbox(tmp_path / "conflict.sqlite3")
    wire = parts[1].to_wire_dict()
    wire["framePartition"]["sequenceNumber"] = "4"
    conflict = TechnicalObservationEvent.model_validate(wire)
    await conflict_box.enqueue(conflict)
    with pytest.raises(OutboxConflict):
        await conflict_box.enqueue_many(parts)
    assert [p.event.event_id for p in await conflict_box.ready()] == [conflict.event_id]
