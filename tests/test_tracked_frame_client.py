import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_processing_worker import REGION_ID, FakeDetector, FakeStream, _configuration, _frame

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences
from smartsite_ai.domain.tracked_raw_receipt import TrackedRawReceipt
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxConflict, OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker

CAMERA = "00000000-0000-4000-8000-000000000001"
GENERATION = "00000000-0000-4000-8000-000000000002"
SESSION = "00000000-0000-4000-8000-000000000003"
FRAME = "00000000-0000-4000-8000-000000000004"
EVENT = "00000000-0000-4000-8000-000000000005"
SEGMENT = "00000000-0000-4000-8000-000000000006"
SCOPE = "a" * 64
CREDENTIAL = WorkerGenerationCredential.model_validate(
    {
        "schemaVersion": "1.0.0",
        "cameraId": CAMERA,
        "workerGenerationId": GENERATION,
        "streamSessionId": SESSION,
        "secret": "b" * 43,
        "expiresAt": "2030-01-01T00:00:00.000Z",
    }
)
RESULT = {
    "kind": "ADMITTED_TRACKED_FRAME",
    "scopeSha256": SCOPE,
    "sequenceNumber": "5",
    "subjects": [{"segmentId": SEGMENT, "eventId": EVENT, "personObservationIndex": 0}],
    "retiredSegmentIds": [],
}


async def read(client):
    return await client.read_tracked_frame_result(
        CREDENTIAL,
        FRAME,
        scope_sha256=SCOPE,
        sequence_number="5",
        subjects={SEGMENT: (EVENT, 0)},
        retired_segment_ids=set(),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("status", [200, 404])
async def test_tracked_frame_get_is_scoped_and_requires_both_credentials(status):
    def handle(request):
        assert request.method == "GET"
        assert request.url.path == (
            f"/api/v1/integrations/ai/cameras/{CAMERA}/generations/{GENERATION}"
            f"/sessions/{SESSION}/tracked-frames/{FRAME}/result"
        )
        assert request.headers["authorization"] == "Bearer synthetic-service"
        assert request.headers["x-worker-generation-secret"] == "b" * 43
        assert not request.url.query
        return httpx.Response(status, json=RESULT)

    async with BackendClient(
        "http://127.0.0.1/api/v1/integrations/ai/events",
        "synthetic-service",
        transport=httpx.MockTransport(handle),
    ) as client:
        result = await read(client)
        if status == 404:
            assert result is None
        else:
            assert result.sequence_number == "5"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"kind": "RAW_STORED"},
        {**RESULT, "sequenceNumber": "6"},
        pytest.param("x" * (1024 * 1024 + 1), id="oversized"),
    ],
)
async def test_invalid_or_unrelated_success_never_advances_tracking(body):
    async with BackendClient(
        "http://127.0.0.1",
        "synthetic-service",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    ) as client:
        with pytest.raises(ValueError, match="Tracked frame result"):
            await read(client)


@pytest.mark.anyio
async def test_redirect_does_not_reissue_worker_credentials():
    calls = []

    def handle(request):
        calls.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://other.invalid/result"})

    async with BackendClient(
        "http://127.0.0.1",
        "synthetic-service",
        transport=httpx.MockTransport(handle),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await read(client)
    assert calls == ["127.0.0.1"]


@pytest.mark.anyio
async def test_persists_only_current_seal_and_recovers_exact_predecessor(tmp_path):
    path = tmp_path / "outbox.sqlite"
    outbox = SqliteEventOutbox(path)
    status, body = 404, RESULT
    sequence = "18446744073709551614"

    async with BackendClient(
        "http://127.0.0.1",
        "synthetic-service",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body)),
    ) as client:

        async def refresh(store, seq, subjects):
            return await store.refresh_tracked_frame(
                client,
                CREDENTIAL,
                FRAME,
                scope_sha256=SCOPE,
                sequence_number=seq,
                subjects=subjects,
                retired_segment_ids=set(),
            )

        assert not await refresh(outbox, sequence, {SEGMENT: (EVENT, 0)})
        assert await outbox.read_tracked_predecessor(SCOPE, sequence) is None
        status, body = 200, {**RESULT, "sequenceNumber": sequence}
        assert await refresh(outbox, sequence, {SEGMENT: (EVENT, 0)})

        restarted = SqliteEventOutbox(path)
        previous = await restarted.read_tracked_predecessor(SCOPE, sequence)
        assert [(s.segment_id, s.event_id, s.person_index) for s in previous.subjects] == [
            (SEGMENT, EVENT, 0)
        ]
        assert await restarted.read_tracked_predecessor("b" * 64, sequence) is None
        assert await restarted.read_tracked_predecessor(SCOPE, str(int(sequence) - 1)) is None
        assert await refresh(restarted, sequence, {SEGMENT: (EVENT, 0)})

        body = {**body, "subjects": [{**RESULT["subjects"][0], "personObservationIndex": 1}]}
        with pytest.raises(OutboxConflict):
            await refresh(restarted, sequence, {SEGMENT: (EVENT, 1)})
        assert await restarted.read_tracked_predecessor(SCOPE, sequence) == previous

        body = {"kind": "RAW_STORED"}
        with pytest.raises(ValueError, match="Tracked frame result"):
            await refresh(restarted, sequence, {SEGMENT: (EVENT, 0)})
        assert await restarted.read_tracked_predecessor(SCOPE, sequence) == previous

        next_sequence = str(int(sequence) + 1)
        body = {**RESULT, "sequenceNumber": next_sequence, "subjects": []}
        assert await refresh(restarted, next_sequence, {})
        body = {**RESULT, "sequenceNumber": sequence}
        assert not await refresh(outbox, sequence, {SEGMENT: (EVENT, 0)})
        assert await outbox.read_tracked_predecessor(SCOPE, sequence) is None
        assert not (await outbox.read_tracked_predecessor(SCOPE, next_sequence)).subjects


@pytest.mark.anyio
async def test_reconcile_pending_seal_after_restart_with_original_generation(tmp_path):
    path = tmp_path / "pending.sqlite"
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(path, clock=lambda: now)
    calls = []
    status = 404

    async def stage(frame, deadline):
        await outbox.stage_tracked_frame(
            CREDENTIAL,
            frame,
            scope_sha256=SCOPE,
            sequence_number="5",
            subjects={SEGMENT: (EVENT, 0)},
            retired_segment_ids=set(),
            expires_at=deadline,
            max_pending=1,
        )

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(status, json=RESULT)

    deadline = now + timedelta(seconds=10)
    await stage(FRAME, deadline)
    await stage(FRAME, deadline)
    with pytest.raises(OutboxConflict):
        await stage(FRAME, deadline + timedelta(seconds=10))
    with pytest.raises(OutboxConflict):
        await stage(EVENT, deadline)

    restarted = SqliteEventOutbox(path, clock=lambda: now)
    other_session = CREDENTIAL.model_copy(update={"stream_session_id": EVENT})
    async with BackendClient(
        "http://127.0.0.1",
        "synthetic-service",
        transport=httpx.MockTransport(handle),
    ) as client:
        assert await restarted.reconcile_tracked_frames(client, other_session, limit=1) == 0
        assert not calls
        assert await restarted.reconcile_tracked_frames(client, CREDENTIAL, limit=1) == 0
        assert await restarted.read_tracked_predecessor(SCOPE, "5") is None
        status = 200
        assert await restarted.reconcile_tracked_frames(client, CREDENTIAL, limit=1) == 1
        assert (await restarted.read_tracked_predecessor(SCOPE, "5")).subjects[0].event_id == EVENT
        assert await restarted.reconcile_tracked_frames(client, CREDENTIAL, limit=1) == 0
        assert len(calls) == 2

        await stage(SEGMENT, deadline)
        now = deadline
        assert await restarted.reconcile_tracked_frames(client, CREDENTIAL, limit=1) == 0
        assert len(calls) == 2


@pytest.mark.anyio
async def test_tracked_send_bytes_and_seal_expectation_are_staged_together(tmp_path):
    path = tmp_path / "tracked.sqlite"
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(path, clock=lambda: now)
    context = EventDeliveryContext.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": CAMERA,
            "workerGenerationId": GENERATION,
            "streamSessionId": SESSION,
            "configurationVersion": 1,
            "viewRevision": 1,
            "bundleSha256": SCOPE,
            "receiptCommandId": EVENT,
            "sourceKind": "LIVE",
        }
    )
    references = TrackedFrameReferences.model_validate(
        {key: value for key, value in RESULT.items() if key != "kind"}
    )
    part = {
        "schemaVersion": "1.5.0",
        "eventId": EVENT,
        "streamSessionId": SESSION,
        "framePartition": {
            "frameId": FRAME,
            "sequenceNumber": "5",
            "partIndex": 0,
            "partCount": 2,
        },
        "observations": [{"type": "PERSON"}],
        "continuity": {
            "coverage": {"status": "COMPLETE"},
            "subjects": RESULT["subjects"],
            "retiredSegments": [],
        },
    }

    second = {
        **part,
        "eventId": "00000000-0000-4000-8000-000000000099",
        "framePartition": {**part["framePartition"], "partIndex": 1},
    }

    async def stage(payload, refs=references, max_bytes=4096):
        parts = (payload, second) if refs is not None or payload is part else (payload,)
        await outbox.enqueue_tracked_parts(
            CREDENTIAL,
            context,
            parts,
            expected_seal=refs,
            expires_at=now + timedelta(seconds=20),
            max_pending=2,
            max_bytes=max_bytes,
        )

    with pytest.raises(ValueError, match="size"):
        await stage(part, max_bytes=1)
    assert not await outbox.ready_tracked_parts(CREDENTIAL, limit=1)
    with pytest.raises(ValueError, match="seal"):
        await stage(part, refs=None)
    assert not await outbox.ready_tracked_parts(CREDENTIAL, limit=1)

    await stage(part)
    await stage(part)
    restarted = SqliteEventOutbox(path, clock=lambda: now)
    ((frame_id, (wire, second_wire)),) = await restarted.ready_tracked_parts(CREDENTIAL, limit=1)
    assert frame_id == FRAME
    assert json.loads(wire) == {
        "schemaVersion": "1.0.0",
        "context": context.model_dump(by_alias=True),
        "event": part,
    }
    assert CREDENTIAL.secret.get_secret_value().encode() not in wire
    assert not await restarted.ready()
    other = CREDENTIAL.model_copy(update={"stream_session_id": EVENT})
    assert not await restarted.ready_tracked_parts(other, limit=1)
    with pytest.raises(OutboxConflict):
        await stage({**part, "cameraExternalId": "changed"})
    assert await restarted.ready_tracked_parts(CREDENTIAL, limit=1) == (
        (FRAME, (wire, second_wire)),
    )

    async with BackendClient(
        "http://127.0.0.1",
        "synthetic-service",
        transport=httpx.MockTransport(lambda _: pytest.fail("seal queried before raw delivery")),
    ) as client:
        assert await restarted.reconcile_tracked_frames(client, CREDENTIAL, limit=1) == 0

    incomplete = {
        **part,
        "eventId": SEGMENT,
        "observations": [],
        "framePartition": {
            **part["framePartition"],
            "frameId": SEGMENT,
            "sequenceNumber": "6",
            "partCount": 1,
        },
        "continuity": {
            "coverage": {"status": "INCOMPLETE", "reason": "INFERENCE_ERROR"},
            "subjects": [],
            "retiredSegments": [],
        },
    }
    await stage(incomplete, refs=None)
    queued = await restarted.ready_tracked_parts(CREDENTIAL, limit=2)
    assert [frame_id for frame_id, _ in queued] == [FRAME, SEGMENT]
    assert json.loads(queued[1][1][0])["event"] == incomplete

    def receipt(body):
        event = json.loads(body)["event"]
        digest = compute_canonical_payload_hash(event)
        return TrackedRawReceipt(
            kind="RAW_STORED",
            eventId=event["eventId"],
            payloadHash=digest,
            bindingHash=context.binding_hash(digest),
        )

    drain_outbox = SqliteEventOutbox(tmp_path / "drain.sqlite", clock=lambda: now)
    await drain_outbox.enqueue_tracked_parts(
        CREDENTIAL,
        context,
        (part, second),
        expected_seal=references,
        expires_at=now + timedelta(seconds=20),
        max_pending=2,
        max_bytes=4096,
    )
    posted = []
    fail_second = True

    def post_part(request):
        nonlocal fail_second
        assert request.method == "POST"
        posted.append(request.content)
        if fail_second and request.content == second_wire:
            fail_second = False
            return httpx.Response(503)
        return httpx.Response(200, json=receipt(request.content).model_dump(by_alias=True))

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(post_part)
    ) as client:
        dispatcher = OutboxDispatcher(drain_outbox, client, generation_credential=CREDENTIAL)
        with pytest.raises(httpx.HTTPStatusError):
            await dispatcher.drain_tracked_once(limit=1)
        assert await drain_outbox.ready_tracked_parts(CREDENTIAL, limit=2) == queued[:1]
        assert posted == [wire, second_wire]

        reopened = SqliteEventOutbox(drain_outbox.path, clock=lambda: now)
        dispatcher = OutboxDispatcher(reopened, client, generation_credential=CREDENTIAL)
        assert await dispatcher.drain_tracked_once(limit=1) == 1
        assert posted == [wire, second_wire, wire, second_wire]
        assert not await reopened.ready_tracked_parts(CREDENTIAL, limit=2)
        assert await reopened.read_tracked_predecessor(SCOPE, "5") is None

    cycle_outbox = SqliteEventOutbox(tmp_path / "cycle.sqlite", clock=lambda: now)
    await cycle_outbox.enqueue_tracked_parts(
        CREDENTIAL,
        context,
        (part, second),
        expected_seal=references,
        expires_at=now + timedelta(seconds=20),
        max_pending=2,
        max_bytes=4096,
    )
    calls = []
    status = 503

    def scheduled_backend(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=RESULT)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json=receipt(request.content).model_dump(by_alias=True))

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(scheduled_backend)
    ) as client:
        default = OutboxDispatcher(cycle_outbox, client, generation_credential=CREDENTIAL)
        assert await default.drain_once() == 0
        assert calls == []
        scheduled = OutboxDispatcher(
            cycle_outbox,
            client,
            generation_credential=CREDENTIAL,
            tracked_batch_limit=1,
        )
        await scheduled.drain_once()
        assert calls == ["POST"]
        assert await cycle_outbox.ready_tracked_parts(CREDENTIAL, limit=1) == queued[:1]

        status = 401
        with pytest.raises(httpx.HTTPStatusError):
            await scheduled.drain_once()
        assert await cycle_outbox.ready_tracked_parts(CREDENTIAL, limit=1) == queued[:1]
        status = 200

        class SealWaitingStream(FakeStream):
            async def get_frame(self):
                if not self.frames:
                    async with asyncio.timeout(2):
                        while await cycle_outbox.read_tracked_predecessor(SCOPE, "5") is None:
                            await asyncio.sleep(0.01)
                return await super().get_frame()

        class NoNewObservation:
            def process(self, *args, **kwargs):
                return None

        stream = SealWaitingStream([_frame(0)])
        configurations = RegionConfigurationStore()
        await configurations.apply(_configuration())
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=FakeDetector(),
            pipeline=NoNewObservation(),
            ppe_region_id=REGION_ID,
            configurations=configurations,
            outbox=cycle_outbox,
            dispatcher=scheduled,
            delivery_interval_seconds=0.01,
        )
        assert (await worker.run()).frames_processed == 1
        assert stream.stopped
        assert calls == ["POST", "POST", "POST", "POST", "GET"]
        assert (await cycle_outbox.read_tracked_predecessor(SCOPE, "5")).subjects[
            0
        ].event_id == EVENT

    first, second_receipt = receipt(wire), receipt(second_wire)
    for receipts in (
        (first,),
        (first, first),
        (first, second_receipt.model_copy(update={"binding_hash": "0" * 64})),
    ):
        with pytest.raises(ValueError, match="receipt"):
            await restarted.confirm_tracked_parts_stored(CREDENTIAL, FRAME, receipts)
        assert await restarted.ready_tracked_parts(CREDENTIAL, limit=2) == queued
    assert not await restarted.confirm_tracked_parts_stored(other, FRAME, (first, second_receipt))
    assert await restarted.confirm_tracked_parts_stored(CREDENTIAL, FRAME, (second_receipt, first))
    assert await restarted.confirm_tracked_parts_stored(CREDENTIAL, FRAME, (first, second_receipt))
    assert await restarted.confirm_tracked_parts_stored(
        CREDENTIAL, SEGMENT, (receipt(queued[1][1][0]),)
    )
    reopened = SqliteEventOutbox(path, clock=lambda: now)
    assert not await reopened.ready_tracked_parts(CREDENTIAL, limit=2)
    assert await reopened.read_tracked_predecessor(SCOPE, "5") is None
    looked_up = []

    def read_seal(request):
        looked_up.append(request.url.path)
        return httpx.Response(200, json=RESULT)

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(read_seal)
    ) as client:
        assert await reopened.reconcile_tracked_frames(client, CREDENTIAL, limit=2) == 1
        assert await reopened.reconcile_tracked_frames(client, CREDENTIAL, limit=2) == 0
    assert len(looked_up) == 1 and looked_up[0].endswith(f"/{FRAME}/result")
    assert (await reopened.read_tracked_predecessor(SCOPE, "5")).subjects[0].event_id == EVENT


@pytest.mark.anyio
@pytest.mark.parametrize("response_mode", ["match", "wrong_binding", "redirect"])
async def test_post_tracked_part_sends_original_bytes_and_requires_matching_receipt(response_mode):
    context = EventDeliveryContext(
        schemaVersion="1.0.0",
        cameraId=CAMERA,
        workerGenerationId=GENERATION,
        streamSessionId=SESSION,
        configurationVersion=1,
        bundleSha256=SCOPE,
        viewRevision=1,
        receiptCommandId=EVENT,
        sourceKind="LIVE",
    )
    event = {"schemaVersion": "1.5.0", "eventId": EVENT, "streamSessionId": SESSION}
    body = json.dumps(
        {"schemaVersion": "1.0.0", "context": context.model_dump(by_alias=True), "event": event}
    ).encode()
    digest = compute_canonical_payload_hash(event)
    binding = context.binding_hash(digest)
    calls = []

    def handle(request):
        calls.append(request.url.host)
        assert request.method == "POST"
        assert request.url.path == "/api/v1/integrations/ai/tracked-parts"
        assert request.content == body
        assert request.headers["authorization"] == "Bearer synthetic-service"
        assert request.headers["x-worker-generation-secret"] == CREDENTIAL.secret.get_secret_value()
        if response_mode == "redirect":
            return httpx.Response(307, headers={"Location": "https://other.invalid/part"})
        return httpx.Response(
            200,
            json={
                "kind": "RAW_STORED",
                "eventId": EVENT,
                "payloadHash": digest,
                "bindingHash": "0" * 64 if response_mode == "wrong_binding" else binding,
            },
        )

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(handle)
    ) as client:
        if response_mode == "match":
            receipt = await client.post_tracked_part(CREDENTIAL, body)
            assert receipt.event_id == EVENT
        elif response_mode == "wrong_binding":
            with pytest.raises(ValueError, match="raw receipt"):
                await client.post_tracked_part(CREDENTIAL, body)
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await client.post_tracked_part(CREDENTIAL, body)
    assert calls == ["127.0.0.1"]


@pytest.mark.anyio
async def test_new_staging_preserves_expired_original_frame_and_deadline(tmp_path):
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(tmp_path / "expiry.sqlite", clock=lambda: now)

    async def stage(frame_id, expiry):
        await outbox.stage_tracked_frame(
            CREDENTIAL,
            frame_id,
            scope_sha256=SCOPE,
            sequence_number="5",
            subjects={SEGMENT: (EVENT, 0)},
            retired_segment_ids=set(),
            expires_at=expiry,
            max_pending=1,
        )

    deadline = now + timedelta(seconds=5)
    await stage(FRAME, deadline)
    with sqlite3.connect(outbox.path) as db:
        original = db.execute(
            "SELECT * FROM tracked_frame_pending WHERE frame_id=?", (FRAME,)
        ).fetchone()
    now = deadline
    await stage(EVENT, now + timedelta(seconds=5))
    with sqlite3.connect(outbox.path) as db:
        preserved = db.execute(
            "SELECT * FROM tracked_frame_pending WHERE frame_id=?", (FRAME,)
        ).fetchone()
    assert preserved == original
    with pytest.raises(OutboxConflict):
        await stage(FRAME, now + timedelta(seconds=10))
    assert await outbox.has_pending_tracked_frame(CREDENTIAL)
