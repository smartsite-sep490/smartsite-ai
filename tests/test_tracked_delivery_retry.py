import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_tracked_frame_client import CREDENTIAL, EVENT, FRAME, RESULT, SCOPE, SEGMENT

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox


async def queue(outbox, now):
    context = EventDeliveryContext.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": CREDENTIAL.camera_id,
            "workerGenerationId": CREDENTIAL.worker_generation_id,
            "streamSessionId": CREDENTIAL.stream_session_id,
            "configurationVersion": 1,
            "viewRevision": 1,
            "bundleSha256": SCOPE,
            "receiptCommandId": EVENT,
            "sourceKind": "LIVE",
        }
    )
    part = {
        "schemaVersion": "1.5.0",
        "eventId": EVENT,
        "streamSessionId": CREDENTIAL.stream_session_id,
        "framePartition": {
            "frameId": FRAME,
            "sequenceNumber": "5",
            "partIndex": 0,
            "partCount": 1,
        },
        "observations": [{"type": "PERSON"}],
        "continuity": {
            "coverage": {"status": "COMPLETE"},
            "subjects": RESULT["subjects"],
            "retiredSegments": [],
        },
    }
    await outbox.enqueue_tracked_parts(
        CREDENTIAL,
        context,
        (part,),
        expected_seal=TrackedFrameReferences.model_validate(
            {key: value for key, value in RESULT.items() if key != "kind"}
        ),
        expires_at=now + timedelta(seconds=20),
        max_pending=2,
        max_bytes=4096,
    )
    return context


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["POST", "GET"])
@pytest.mark.parametrize("failure", [408, 429, 503, "transport"])
async def test_transient_retry_is_durable_due_bounded_and_preserves_original_bytes(
    tmp_path, phase, failure
):
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(tmp_path / "retry.sqlite", clock=lambda: now)
    context = await queue(outbox, now)
    ((_, (original,)),) = await outbox.ready_tracked_parts(CREDENTIAL, limit=1)
    calls = []
    failing = True

    def handle(request):
        calls.append((request.method, request.content))
        if failing and request.method == phase:
            if failure == "transport":
                raise httpx.ConnectError("synthetic disconnect", request=request)
            return httpx.Response(failure)
        if request.method == "GET":
            return httpx.Response(200, json=RESULT)
        event = json.loads(request.content)["event"]
        payload_hash = compute_canonical_payload_hash(event)
        return httpx.Response(
            200,
            json={
                "kind": "RAW_STORED",
                "eventId": event["eventId"],
                "payloadHash": payload_hash,
                "bindingHash": context.binding_hash(payload_hash),
            },
        )

    def dispatcher(current, client):
        return OutboxDispatcher(
            current,
            client,
            generation_credential=CREDENTIAL,
            tracked_batch_limit=1,
            retry_base=timedelta(seconds=2),
            retry_max=timedelta(seconds=3),
        )

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(handle)
    ) as client:
        if phase == "GET":
            await dispatcher(outbox, client).drain_once()
            calls.clear()
        await dispatcher(outbox, client).drain_once()
        assert [method for method, _ in calls] == [phase]
        reopened = SqliteEventOutbox(outbox.path, clock=lambda: now)
        await dispatcher(reopened, client).drain_once()
        assert len(calls) == 1, "restart must retain the future retry time"
        now += timedelta(seconds=1)
        await dispatcher(reopened, client).drain_once()
        assert len(calls) == 1
        now += timedelta(seconds=1)
        await dispatcher(reopened, client).drain_once()
        assert len(calls) == 2
        now += timedelta(seconds=2)
        await dispatcher(reopened, client).drain_once()
        assert len(calls) == 2, "second failure must use the capped three-second delay"
        now += timedelta(seconds=1)
        failing = False
        await dispatcher(reopened, client).drain_once()
        assert len(calls) == 3
        if phase == "POST":
            assert all(body == original for method, body in calls if method == "POST")
            assert await reopened.read_tracked_predecessor(SCOPE, "5") is None
            await dispatcher(reopened, client).drain_once()
        sealed = await reopened.read_tracked_predecessor(SCOPE, "5")
        assert sealed is not None
        assert sealed.subject_references() == {SEGMENT: (EVENT, 0)}


@pytest.mark.anyio
async def test_existing_tracked_database_upgrades_without_losing_wire_or_expiry(tmp_path):
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(tmp_path / "old.sqlite", clock=lambda: now)
    await queue(outbox, now)
    original = await outbox.ready_tracked_parts(CREDENTIAL, limit=1)
    with sqlite3.connect(outbox.path) as db:
        expiry = db.execute("SELECT expires_at FROM tracked_frame_pending").fetchone()[0]
        db.execute("ALTER TABLE tracked_frame_pending DROP COLUMN attempt_count")
        db.execute("ALTER TABLE tracked_frame_pending DROP COLUMN next_attempt_at")
    reopened = SqliteEventOutbox(outbox.path, clock=lambda: now)
    assert await reopened.ready_tracked_parts(CREDENTIAL, limit=1) == original
    await reopened.defer_tracked_frame(
        CREDENTIAL, FRAME, retry_base=timedelta(seconds=2), retry_max=timedelta(seconds=3)
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio,sys; from pathlib import Path; from datetime import datetime; "
            "from smartsite_ai.integrations.outbox import SqliteEventOutbox; "
            "from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential; "
            "outbox=SqliteEventOutbox(Path(sys.argv[1]),"
            "clock=lambda:datetime.fromisoformat(sys.argv[2])); "
            "credential=WorkerGenerationCredential.model_validate_json(sys.argv[3]); "
            "assert asyncio.run(outbox.ready_tracked_parts(credential,limit=1)) == ()",
            str(outbox.path),
            now.isoformat(),
            json.dumps({**CREDENTIAL.model_dump(mode="json", by_alias=True), "secret": "b" * 43}),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 0, child.stderr.decode()
    with sqlite3.connect(outbox.path) as db:
        assert db.execute("SELECT expires_at FROM tracked_frame_pending").fetchone()[0] == expiry
    now += timedelta(seconds=21)
    assert not await reopened.ready_tracked_parts(CREDENTIAL, limit=1)
    assert await reopened.has_pending_tracked_frame(CREDENTIAL)
    with sqlite3.connect(outbox.path) as db:
        stored = db.execute("SELECT delivery_json FROM tracked_frame_pending").fetchone()[0]
    assert tuple(body.encode() for body in json.loads(stored)) == original[0][1]


@pytest.mark.anyio
@pytest.mark.parametrize("phase", ["POST", "GET"])
@pytest.mark.parametrize("status", [401, 403, 409, 422])
async def test_hard_failure_propagates_without_a_retry_schedule(tmp_path, phase, status):
    now = datetime(2026, 11, 8, tzinfo=UTC)
    outbox = SqliteEventOutbox(tmp_path / "hard.sqlite", clock=lambda: now)
    await queue(outbox, now)
    if phase == "GET":
        # Stage lookup-only work using the original expected result; raw intake is separate.
        with sqlite3.connect(outbox.path) as db:
            db.execute("UPDATE tracked_frame_pending SET raw_stored = 1")
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(status)

    async with BackendClient(
        "http://127.0.0.1", "synthetic-service", transport=httpx.MockTransport(handle)
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox, client, generation_credential=CREDENTIAL, tracked_batch_limit=1
        )
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError) as error:
                await dispatcher.drain_once()
            assert error.value.response.status_code == status
        assert calls == [phase, phase]
        assert await outbox.read_tracked_predecessor(SCOPE, "5") is None
