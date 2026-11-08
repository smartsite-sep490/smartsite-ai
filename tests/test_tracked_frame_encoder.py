import asyncio
import hashlib
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from test_mf05_mf06_pipeline import batch, configuration, detection, make_pipeline
from test_tracked_frame_client import CREDENTIAL, EVENT

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import PpeObservation
from smartsite_ai.domain.tracked_frame_result import TrackedFrameResult
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.evidence.models import EvidenceBindingError
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.integrations.tracked_capture import stage_tracked_capture
from smartsite_ai.pipelines.ppe_temporal import ConfirmedPpeCandidate
from smartsite_ai.pipelines.tracked_frame_encoder import (
    encode_complete_tracked_frame,
    encode_incomplete_tracked_frame,
)

SCOPE = "a" * 64
PROFILE = "b" * 64


def test_continue_uses_only_the_exact_admitted_subject():
    pipeline = make_pipeline()
    person = detection("person", (0.1, 0.1, 0.3, 0.8))
    config = configuration()
    first = pipeline.observe_frame(
        batch(1, person),
        region_configuration=config,
        event_id="00000000-0000-4000-8000-000000000019",
    )
    next_frame = pipeline.observe_frame(
        batch(2, person),
        region_configuration=config,
        event_id="00000000-0000-4000-8000-000000000020",
    )
    assert first is not None and next_frame is not None
    start, expected = encode_complete_tracked_frame(
        first, scope_sha256=SCOPE, profile_sha256=PROFILE
    )
    with pytest.raises(ValueError, match="admitted predecessor"):
        encode_complete_tracked_frame(next_frame, scope_sha256=SCOPE, profile_sha256=PROFILE)

    admitted = TrackedFrameResult.model_validate(
        {"kind": "ADMITTED_TRACKED_FRAME", **expected.model_dump(by_alias=True)}
    )
    wire, _ = encode_complete_tracked_frame(
        next_frame, scope_sha256=SCOPE, profile_sha256=PROFILE, predecessor=admitted
    )
    claim = wire[0]["continuity"]["subjects"][0]
    assert claim["link"] == {
        "kind": "CONTINUE",
        "previousEventId": start[0]["eventId"],
        "previousPersonObservationIndex": start[0]["continuity"]["subjects"][0][
            "personObservationIndex"
        ],
    }
    assert wire[0]["observations"][claim["personObservationIndex"]]["type"] == "PERSON"
    with pytest.raises(ValueError, match="admitted predecessor"):
        encode_complete_tracked_frame(
            next_frame, scope_sha256="c" * 64, profile_sha256=PROFILE, predecessor=admitted
        )
    with pytest.raises(ValueError, match="admitted predecessor"):
        encode_complete_tracked_frame(
            replace(next_frame, source_sequence_number=3),
            scope_sha256=SCOPE,
            profile_sha256=PROFILE,
            predecessor=admitted,
        )

    empty = pipeline.observe_frame(
        batch(3), region_configuration=config, event_id="00000000-0000-4000-8000-000000000021"
    )
    assert empty is not None
    parts, refs = encode_complete_tracked_frame(empty, scope_sha256=SCOPE, profile_sha256=PROFILE)
    assert parts[0]["observations"] == []
    assert refs.retired_segment_ids == (claim["segmentId"],)


def test_incomplete_capture_retains_observations_but_emits_no_continuity_claims():
    pipeline = make_pipeline()
    config = configuration()
    person = detection("person", (0.1, 0.1, 0.3, 0.8))
    for sequence, detections in enumerate(((person,), ()), 1):
        frame = pipeline.observe_frame(
            batch(sequence, *detections),
            region_configuration=config,
            event_id=f"00000000-0000-4000-8000-{sequence:012d}",
        )
        assert frame is not None
        expected = (
            frame.to_partitioned_events(sequence_number=sequence)[0].to_wire_dict()["observations"]
            if detections
            else []
        )
        assert frame.subject_segments if detections else frame.retired_segments
        (event,) = encode_incomplete_tracked_frame(
            frame, profile_sha256=PROFILE, reason="AMBIGUOUS_PROCESSING"
        )
        assert event["observations"] == expected
        assert event["framePartition"]["sequenceNumber"] == str(sequence)
        assert event["continuity"] == {
            "schemaVersion": "1.0.0",
            "profileSha256": PROFILE,
            "coverage": {"status": "INCOMPLETE", "reason": "AMBIGUOUS_PROCESSING"},
            "subjects": [],
            "retiredSegments": [],
        }


@pytest.mark.parametrize("reason", ["SOURCE_UNAVAILABLE", "DECODE_ERROR"])
def test_missing_source_is_not_a_captured_incomplete_frame(reason):
    frame = make_pipeline().observe_frame(
        batch(1),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000021",
    )
    assert frame is not None
    with pytest.raises(ValueError, match="captured-frame coverage reason"):
        encode_incomplete_tracked_frame(frame, profile_sha256=PROFILE, reason=reason)


@pytest.mark.anyio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_producer_keeps_people_and_only_confirmed_missing_ppe(tmp_path, confirmed):
    pipeline = make_pipeline()
    person = detection("person", (0.1, 0.1, 0.3, 0.8))
    raw = pipeline.observe_frame(
        batch(1, person), region_configuration=configuration(), event_id=EVENT
    )
    assert raw is not None
    track_id = raw.subject_segments[0].track_id
    region_id = configuration().regions[0].region_id
    missing = PpeObservation(
        type="PPE",
        trackId=track_id,
        regionId=region_id,
        geometryVersion=1,
        ppeItem="HARD_HAT",
        status="MISSING",
    )
    frame = replace(
        raw,
        stream_session_id=CREDENTIAL.stream_session_id,
        observations=(*[obs for obs in raw.observations if obs.type != "PPE"], missing),
    )
    now = datetime.now(UTC)
    candidates = (
        (
            ConfirmedPpeCandidate(
                stream_id="stream-01",
                session_id=UUID(frame.stream_session_id),
                track_id=track_id,
                ppe_item="HARD_HAT",
                first_seen_at=now,
                confirmed_at=now,
                region_id=region_id,
                geometry_version=1,
            ),
        )
        if confirmed
        else ()
    )
    context = EventDeliveryContext(
        schemaVersion="1.0.0",
        cameraId=CREDENTIAL.camera_id,
        workerGenerationId=CREDENTIAL.worker_generation_id,
        streamSessionId=CREDENTIAL.stream_session_id,
        configurationVersion=1,
        bundleSha256=SCOPE,
        viewRevision=1,
        receiptCommandId=EVENT,
        sourceKind="LIVE",
    )
    site = "00000000-0000-4000-8000-000000000071"
    outbox = SqliteEventOutbox(tmp_path / "producer.sqlite")
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    publisher = LocalEvidencePublisher(root_dir=evidence_root, bind_content_hash=True)
    source = FrameEnvelope(
        stream_id="stream-01",
        session_id=UUID(frame.stream_session_id),
        camera_external_id=frame.camera_external_id,
        captured_at=datetime.fromisoformat(frame.captured_at),
        width=frame.frame_dimensions.width,
        height=frame.frame_dimensions.height,
        sequence_number=1,
        payload=b"\0" * (frame.frame_dimensions.width * frame.frame_dimensions.height * 3),
    )
    if confirmed:
        with pytest.raises(EvidenceBindingError, match="partition sequence"):
            await stage_tracked_capture(
                outbox,
                frame,
                context,
                CREDENTIAL,
                site_id=site,
                profile_sha256=PROFILE,
                confirmed_ppe_candidates=candidates,
                expires_at=now + timedelta(seconds=30),
                max_pending=2,
                max_bytes=16000,
                source_frame=source.model_copy(update={"sequence_number": 2}),
                evidence_publisher=publisher,
            )
        assert not await outbox.ready_tracked_parts(CREDENTIAL, limit=1)
        assert not list(evidence_root.iterdir())
    staged = await stage_tracked_capture(
        outbox,
        frame,
        context,
        CREDENTIAL,
        site_id=site,
        profile_sha256=PROFILE,
        confirmed_ppe_candidates=candidates,
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
        max_pending=2,
        max_bytes=16000,
        source_frame=source,
        evidence_publisher=publisher,
    )
    assert staged.scope_sha256 == compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": site,
            "profileSha256": PROFILE,
        }
    )
    ((frame_id, (body,)),) = await outbox.ready_tracked_parts(CREDENTIAL, limit=1)
    assert frame_id == EVENT
    part = json.loads(body)["event"]
    assert any(obs["type"] == "PERSON" for obs in part["observations"])
    assert (
        any(obs["type"] == "PPE" and obs["status"] == "MISSING" for obs in part["observations"])
        is confirmed
    )
    if confirmed:
        (reference,) = part["evidence"]
        event_id = part["eventId"]
        assert reference["uri"] == f"local://evidence/{frame.stream_session_id}/1/{event_id}.jpg"
        jpeg = (evidence_root / f"{frame.stream_session_id}_1_{event_id}.jpg").read_bytes()
        assert jpeg.startswith(b"\xff\xd8")
        assert reference["sha256"] == hashlib.sha256(jpeg).hexdigest()
    else:
        assert part["evidence"] == []
        assert not list(evidence_root.iterdir())
    assert not await outbox.ready()

    following = pipeline.observe_frame(
        batch(2, person),
        region_configuration=configuration(),
        event_id="00000000-0000-4000-8000-000000000072",
    )
    assert following is not None
    with pytest.raises(ValueError, match="admitted predecessor"):
        await stage_tracked_capture(
            outbox,
            replace(following, stream_session_id=frame.stream_session_id),
            context,
            CREDENTIAL,
            site_id=site,
            profile_sha256=PROFILE,
            confirmed_ppe_candidates=(),
            expires_at=now + timedelta(seconds=30),
            max_pending=2,
            max_bytes=16000,
        )
    assert await outbox.ready_tracked_parts(CREDENTIAL, limit=2) == ((frame_id, (body,)),)


@pytest.mark.anyio
async def test_staging_cancellation_waits_for_the_sqlite_commit(tmp_path):
    pipeline = make_pipeline()
    observed = pipeline.observe_frame(
        batch(1, detection("person", (0.1, 0.1, 0.3, 0.8))),
        region_configuration=configuration(),
        event_id=EVENT,
    )
    frame = replace(observed, stream_session_id=CREDENTIAL.stream_session_id)
    context = EventDeliveryContext(
        schemaVersion="1.0.0",
        cameraId=CREDENTIAL.camera_id,
        workerGenerationId=CREDENTIAL.worker_generation_id,
        streamSessionId=CREDENTIAL.stream_session_id,
        configurationVersion=1,
        bundleSha256=SCOPE,
        viewRevision=1,
        receiptCommandId=EVENT,
        sourceKind="LIVE",
    )
    entered = threading.Event()
    release = threading.Event()

    class HeldOutbox(SqliteEventOutbox):
        def _stage_tracked_frame_sync(self, *args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Fixture failed to release SQLite staging")
            return super()._stage_tracked_frame_sync(*args)

    outbox = HeldOutbox(tmp_path / "cancel.sqlite")
    committed = []
    task = asyncio.create_task(
        stage_tracked_capture(
            outbox,
            frame,
            context,
            CREDENTIAL,
            site_id="00000000-0000-4000-8000-000000000071",
            profile_sha256=PROFILE,
            confirmed_ppe_candidates=(),
            expires_at=datetime.now(UTC) + timedelta(seconds=20),
            max_pending=2,
            max_bytes=16000,
            on_staged=committed.append,
        )
    )
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert committed == [1]
    assert len(await outbox.ready_tracked_parts(CREDENTIAL, limit=1)) == 1
