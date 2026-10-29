import asyncio
import hashlib

import pytest
from pydantic import ValidationError
from test_hash_bound_evidence import frame_and_event

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.evidence.local_publisher import EvidenceBindingError, LocalEvidencePublisher
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.pipelines.ppe_temporal import filter_event_for_delivery


def payload():
    frame, legacy = frame_and_event()
    wire = legacy.to_wire_dict()
    wire.update(
        schemaVersion="1.4.0",
        framePartition={
            "frameId": "aaaaaaaa-1111-4111-8111-111111111111",
            "sequenceNumber": "18446744073709551615",
            "partIndex": 0,
            "partCount": 2,
        },
    )
    wire["observations"].insert(0, {"type": "PERSON", "trackId": 1})
    return frame, wire


def test_explicit_partition_model_preserves_exact_uint64_and_legacy_shape():
    _, wire = payload()
    event = TechnicalObservationEvent.model_validate(wire)
    assert event.to_wire_dict() == wire
    assert (
        TechnicalObservationEvent.model_validate_json(
            event.model_dump_json(by_alias=True, exclude_none=True)
        )
        == event
    )
    frame, legacy = frame_and_event()
    assert "framePartition" not in legacy.to_wire_dict()
    assert legacy.schema_version == "1.0.0"
    assert frame.sequence_number == 1


@pytest.mark.parametrize(
    "change",
    [
        {"partIndex": -1},
        {"partIndex": 2},
        {"partIndex": True},
        {"partCount": 0},
        {"partCount": 1025},
        {"sequenceNumber": "18446744073709551616"},
        {"sequenceNumber": "01"},
        {"sequenceNumber": 1},
        {"sequenceNumber": "-1"},
        {"frameId": "AAAAAAAA-1111-4111-8111-111111111111"},
        {"frameId": "not-uuid"},
        {"unknown": "private"},
    ],
)
def test_invalid_partition_metadata_is_rejected(change):
    _, wire = payload()
    wire["framePartition"].update(change)
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(wire)


@pytest.mark.parametrize("case", ["missing", "legacy", "orphan", "duplicate", "unhashed"])
def test_partition_version_person_and_evidence_boundaries(case):
    _, wire = payload()
    if case == "missing":
        del wire["framePartition"]
    elif case == "legacy":
        wire["schemaVersion"] = "1.3.0"
    elif case == "orphan":
        wire["observations"].pop(0)
    elif case == "duplicate":
        wire["observations"].append({"type": "PERSON", "trackId": 1})
    else:
        wire["evidence"] = [{"kind": "FRAME", "uri": "local://synthetic"}]
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(wire)


def test_partition_metadata_survives_real_jpeg_publication_and_sqlite_replay(tmp_path):
    frame, wire = payload()
    wire["framePartition"]["sequenceNumber"] = "1"
    event = TechnicalObservationEvent.model_validate(wire)
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)

    async def round_trip():
        published = await publisher.publish(frame=frame, event=event)
        outbox = SqliteEventOutbox(tmp_path / "partition.sqlite3")
        await outbox.enqueue(published)
        pending = await outbox.ready()
        return published, pending[0].event

    published, replay = asyncio.run(round_trip())
    assert replay == published
    assert published.schema_version == "1.4.0"
    assert published.to_wire_dict()["framePartition"] == wire["framePartition"]
    assert (
        published.evidence[0].sha256
        == hashlib.sha256(next(tmp_path.glob("*.jpg")).read_bytes()).hexdigest()
    )
    assert published.compute_payload_hash() == replay.compute_payload_hash()


def test_partition_source_sequence_mismatch_writes_no_image(tmp_path):
    frame, wire = payload()
    event = TechnicalObservationEvent.model_validate(wire)
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    with pytest.raises(EvidenceBindingError):
        asyncio.run(publisher.publish(frame=frame, event=event))
    assert list(tmp_path.iterdir()) == []


def test_delivery_filter_preserves_partition_metadata():
    _, wire = payload()
    event = TechnicalObservationEvent.model_validate(wire)
    filtered = filter_event_for_delivery(event)
    assert filtered is not None
    assert filtered.to_wire_dict() == wire


def test_partition_publisher_requires_content_binding_before_any_write(tmp_path):
    frame, wire = payload()
    wire["framePartition"]["sequenceNumber"] = "1"
    event = TechnicalObservationEvent.model_validate(wire)
    with pytest.raises(EvidenceBindingError):
        asyncio.run(LocalEvidencePublisher(root_dir=tmp_path).publish(frame=frame, event=event))
    assert list(tmp_path.iterdir()) == []
