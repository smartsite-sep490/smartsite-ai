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


@pytest.mark.anyio
async def test_scope_change_during_predecessor_wait_aborts_before_evidence_publish(tmp_path):
    from dataclasses import replace

    from test_tracked_encoder import frame as make_obs_frame

    from smartsite_ai.domain.tracked_observations import TrackedScopeTransitionRequired
    from smartsite_ai.tracking.models import SubjectSegment

    now = datetime.now(UTC)
    f0 = make_obs_frame()
    _, context, credential, *_ = packet()
    context = context.model_copy(
        update={
            "stream_session_id": f0.stream_session_id,
            "configuration_version": f0.camera_configuration_version or 1,
            "source_kind": "LIVE",
        }
    )
    f0 = replace(f0, camera_configuration_version=context.configuration_version)
    credential = credential.model_copy(update={"stream_session_id": f0.stream_session_id})
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite")
    publisher = LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True)
    site_refs = await stage_tracked_capture(
        outbox,
        f0,
        context,
        credential,
        site_id=SITE,
        profile_sha256="b" * 64,
        confirmed_ppe_candidates=(),
        expires_at=now + timedelta(seconds=30),
        max_pending=4,
        max_bytes=200_000,
    )

    obs1, confirmed1 = observed()
    seg0 = f0.subject_segments[0].segment_id
    obs1 = replace(
        obs1,
        event_id="00000000-0000-4000-8000-000000000101",
        source_sequence_number=1,
        captured_at=(now + timedelta(seconds=1)).isoformat(),
        camera_configuration_version=f0.camera_configuration_version,
        subject_segments=(
            SubjectSegment(track_id=1, segment_id=seg0, previous_sequence_number=0, reason=None),
        ),
    )
    confirmed1 = (replace(confirmed1[0], confirmed_at=now + timedelta(seconds=1)),)
    src1 = FrameEnvelope(
        stream_id="stream-01",
        session_id=UUID(obs1.stream_session_id),
        camera_external_id=obs1.camera_external_id,
        captured_at=now + timedelta(seconds=1),
        width=obs1.frame_dimensions.width,
        height=obs1.frame_dimensions.height,
        sequence_number=1,
        payload=bytes([30, 50, 80]) * obs1.frame_dimensions.width * obs1.frame_dimensions.height,
    )
    transitioned = False

    async def check_scope():
        if transitioned:
            raise TrackedScopeTransitionRequired("scope changed during predecessor wait")

    async def trigger():
        nonlocal transitioned
        await asyncio.sleep(0.03)
        transitioned = True
        await asyncio.sleep(0.03)
        conn = outbox._connect()
        try:
            conn.execute(
                "INSERT INTO tracked_frame_head(scope_sha256, frame_id, result_json)"
                " VALUES (?, ?, ?)",
                (
                    site_refs.scope_sha256,
                    f0.event_id,
                    json.dumps(
                        {
                            "kind": "ADMITTED_TRACKED_FRAME",
                            "scopeSha256": site_refs.scope_sha256,
                            "sequenceNumber": "0",
                            "subjects": [
                                {
                                    "segmentId": str(seg0),
                                    "eventId": f0.event_id,
                                    "personObservationIndex": 0,
                                }
                            ],
                            "retiredSegmentIds": [],
                        }
                    ),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    bg = asyncio.create_task(trigger())
    try:
        with pytest.raises(TrackedScopeTransitionRequired, match="scope changed"):
            await stage_tracked_capture(
                outbox,
                obs1,
                context,
                credential,
                site_id=SITE,
                profile_sha256="b" * 64,
                confirmed_ppe_candidates=confirmed1,
                expires_at=now + timedelta(seconds=30),
                max_pending=4,
                max_bytes=200_000,
                source_frame=src1,
                evidence_publisher=publisher,
                predecessor_wait_seconds=1.0,
                poll_interval_seconds=0.01,
                before_enqueue=check_scope,
            )
    finally:
        await bg
    assert list(tmp_path.glob("*.jpg")) == []
    assert [fid for fid, _ in await outbox.ready_tracked_parts(credential, limit=10)] == [
        f0.event_id.lower()
    ]


@pytest.mark.anyio
async def test_predecessor_and_capacity_waits_do_not_outlive_credential_or_lease(tmp_path):
    from dataclasses import replace

    from test_tracked_encoder import frame as make_obs_frame

    from smartsite_ai.tracking.models import SubjectSegment

    now = datetime.now(UTC)
    f0 = make_obs_frame()
    _, context, credential, *_ = packet()
    short_expiry = (now + timedelta(milliseconds=80)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    context = context.model_copy(
        update={
            "stream_session_id": f0.stream_session_id,
            "configuration_version": f0.camera_configuration_version or 1,
            "source_kind": "LIVE",
        }
    )
    f0 = replace(f0, camera_configuration_version=context.configuration_version)
    expiring_credential = credential.model_copy(
        update={"stream_session_id": f0.stream_session_id, "expires_at": short_expiry}
    )
    outbox = SqliteEventOutbox(tmp_path / "lease.sqlite")
    seg0 = f0.subject_segments[0].segment_id
    obs1 = replace(
        f0,
        event_id="00000000-0000-4000-8000-000000000102",
        source_sequence_number=1,
        captured_at=(now + timedelta(seconds=1)).isoformat(),
        camera_configuration_version=context.configuration_version,
        subject_segments=(
            SubjectSegment(track_id=1, segment_id=seg0, previous_sequence_number=0, reason=None),
        ),
    )

    started = asyncio.get_running_loop().time()
    with pytest.raises((RuntimeError, TimeoutError)):
        await stage_tracked_capture(
            outbox,
            obs1,
            context,
            expiring_credential,
            site_id=SITE,
            profile_sha256="b" * 64,
            confirmed_ppe_candidates=(),
            expires_at=now + timedelta(seconds=30),
            max_pending=2,
            max_bytes=200_000,
            predecessor_wait_seconds=5.0,
            poll_interval_seconds=0.01,
        )
    assert asyncio.get_running_loop().time() - started < 0.5

    # Capacity retry is also bounded by remaining frame lease and never extends it.
    from smartsite_ai.integrations.outbox import TrackedOutboxCapacityReached

    await stage_tracked_capture(
        outbox,
        f0,
        context,
        credential,
        site_id=SITE,
        profile_sha256="b" * 64,
        confirmed_ppe_candidates=(),
        expires_at=now + timedelta(seconds=30),
        max_pending=1,
        max_bytes=200_000,
    )
    obs1_independent = replace(
        f0,
        event_id="00000000-0000-4000-8000-000000000103",
        source_sequence_number=1,
        captured_at=(now + timedelta(seconds=1)).isoformat(),
        camera_configuration_version=context.configuration_version,
        subject_segments=(
            SubjectSegment(
                track_id=1,
                segment_id=f0.subject_segments[0].segment_id,
                previous_sequence_number=None,
                reason="INITIAL",
            ),
        ),
    )
    cap_started = asyncio.get_running_loop().time()
    with pytest.raises((TrackedOutboxCapacityReached, TimeoutError)):
        await stage_tracked_capture(
            outbox,
            obs1_independent,
            context,
            credential,
            site_id=SITE,
            profile_sha256="b" * 64,
            confirmed_ppe_candidates=(),
            expires_at=datetime.now(UTC) + timedelta(milliseconds=120),
            max_pending=1,
            max_bytes=200_000,
            predecessor_wait_seconds=5.0,
            poll_interval_seconds=0.02,
        )
    assert asyncio.get_running_loop().time() - cap_started < 0.5
