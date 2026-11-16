import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from test_tracked_encoder import frame
from test_tracked_transport import packet

from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.integrations.tracked_capture import stage_tracked_capture
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate

NOW = datetime(2026, 11, 17, tzinfo=UTC)
SITE = "00000000-0000-4000-8000-000000000077"


def observed():
    value = replace(frame(), captured_at=NOW.isoformat(), camera_configuration_version=1)
    gate = TemporalPpeCandidateGate(confirmation_frames=1)
    confirmed = gate.update(
        stream_id="stream-01",
        session_id=UUID(value.stream_session_id),
        observed_at=NOW,
        active_track_ids=(1,),
        observations=(value.observations[1],),
        camera_configuration_version=1,
    )
    return value, confirmed


async def stage(outbox, **kwargs):
    value, confirmed = observed()
    _, context, credential, *_ = packet()
    return await stage_tracked_capture(
        outbox,
        value,
        context,
        credential,
        site_id=SITE,
        profile_sha256="b" * 64,
        confirmed_ppe_candidates=confirmed,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=100_000,
        **kwargs,
    )


@pytest.mark.anyio
async def test_tracked_capture_retains_sha_bound_original_frame_image(tmp_path):
    value, _ = observed()
    source = FrameEnvelope(
        stream_id="stream-01",
        session_id=UUID(value.stream_session_id),
        camera_external_id=value.camera_external_id,
        captured_at=NOW,
        width=value.frame_dimensions.width,
        height=value.frame_dimensions.height,
        sequence_number=0,
        payload=bytes([30, 50, 80]) * value.frame_dimensions.width * value.frame_dimensions.height,
    )
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    refs = await stage(outbox, source_frame=source, evidence_publisher=publisher)
    _, _, credential, *_ = packet()
    ((_, (wire,)),) = await outbox.ready_tracked_parts(credential, limit=1)
    event = json.loads(wire)["event"]
    assert len(event["evidence"]) == 1
    images = list(tmp_path.glob("*.jpg"))
    assert len(images) == 1
    assert hashlib.sha256(images[0].read_bytes()).hexdigest() == event["evidence"][0]["sha256"]
    assert event["evidence"][0]["uri"].endswith(f"/{event['eventId']}.jpg")
    assert refs.subjects[0].event_id == event["eventId"]
    assert [item["type"] for item in event["observations"]] == ["PERSON", "PPE"]


@pytest.mark.anyio
async def test_guard_failure_before_enqueue_does_not_commit_an_unapproved_context(tmp_path):
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)

    async def changed():
        raise RuntimeError("configuration changed")

    with pytest.raises(RuntimeError, match="configuration changed"):
        await stage(outbox, before_enqueue=changed)
    _, _, credential, *_ = packet()
    assert not await outbox.has_pending_tracked_frame(credential)


@pytest.mark.anyio
async def test_cancellation_waits_for_owned_persistence_before_reporting_commit():
    entered, release = asyncio.Event(), asyncio.Event()
    staged = []

    class ControlledOutbox:
        async def enqueue_tracked_parts(self, *_args, **_kwargs):
            entered.set()
            await release.wait()

    task = asyncio.create_task(stage(ControlledOutbox(), on_staged=staged.append))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and staged == []
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert staged == [1]
