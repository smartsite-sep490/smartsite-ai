import asyncio
import hashlib
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import SqliteEventOutbox


def frame_and_event(version="1.0.0"):
    frame = FrameEnvelope(
        stream_id="hash-test",
        session_id=UUID("11111111-1111-4111-8111-111111111111"),
        camera_external_id="hash-camera",
        captured_at=datetime(2026, 10, 22, tzinfo=UTC),
        width=4,
        height=4,
        sequence_number=1,
        payload=bytes([30, 20, 10] * 16),
    )
    event = TechnicalObservationEvent.create(
        event_id="22222222-2222-4222-8222-222222222222",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at.isoformat(),
        frame_dimensions={"width": 4, "height": 4},
        observations=[
            {
                "type": "ZONE_ENTRY",
                "trackId": 1,
                "regionId": "33333333-3333-4333-8333-333333333333",
                "geometryVersion": 1,
            }
        ],
        schema_version=version,
    )
    return frame, event


@pytest.mark.parametrize(("version", "expected"), [("1.0.0", "1.2.0"), ("1.1.0", "1.3.0")])
def test_publisher_binds_actual_bytes_and_outbox_round_trip(tmp_path, version, expected):
    frame, event = frame_and_event(version)
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    published = asyncio.run(publisher.publish(frame=frame, event=event))
    assert published.schema_version == expected
    assert published.event_id == event.event_id
    assert published.observations == event.observations
    jpeg = next(tmp_path.glob("*.jpg")).read_bytes()
    assert published.evidence[0].sha256 == hashlib.sha256(jpeg).hexdigest()
    wire = published.to_wire_dict()
    assert (
        TechnicalObservationEvent.model_validate(wire).compute_payload_hash()
        == published.compute_payload_hash()
    )
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")

    async def round_trip():
        await outbox.initialize()
        await outbox.enqueue(published)
        return await outbox.ready()

    pending = asyncio.run(round_trip())
    assert pending[0].event == published


def test_legacy_publisher_does_not_change_wire_version(tmp_path):
    frame, event = frame_and_event()
    published = asyncio.run(
        LocalEvidencePublisher(root_dir=tmp_path).publish(frame=frame, event=event)
    )
    assert published.schema_version == "1.0.0"
    assert "sha256" not in published.to_wire_dict()["evidence"][0]


@pytest.mark.parametrize("version", ["1.2.0", "1.3.0"])
def test_hash_bound_version_rejects_unhashed_reference(version):
    _, event = frame_and_event()
    wire = event.to_wire_dict()
    wire.update(schemaVersion=version, evidence=[{"kind": "FRAME", "uri": "local://test"}])
    with pytest.raises(ValidationError, match="hash-bound evidence requires sha256"):
        TechnicalObservationEvent.model_validate(wire)


def test_legacy_version_rejects_hash_extension():
    _, event = frame_and_event()
    wire = event.to_wire_dict()
    wire["evidence"] = [{"kind": "FRAME", "uri": "local://test", "sha256": "a" * 64}]
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(wire)
