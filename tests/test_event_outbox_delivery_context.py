from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import (
    OutboxConflict,
    OutboxDeliveryBlocked,
    OutboxDispatcher,
    OutboxEnqueueOutcome,
    SqliteEventOutbox,
)
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential


def _context(**changes):
    try:
        api = importlib.import_module("smartsite_ai.domain.event_delivery_context")
    except ModuleNotFoundError:
        pytest.fail("Original execution context must have a strict immutable model")
    value = {
        "schemaVersion": "1.0.0",
        "cameraId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "workerGenerationId": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "streamSessionId": "22222222-2222-4222-8222-222222222222",
        "configurationVersion": 2,
        "bundleSha256": "a" * 64,
        "viewRevision": 1,
        "receiptCommandId": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        "sourceKind": "LIVE",
        **changes,
    }
    return api.EventDeliveryContext.model_validate(value)


def _event(event_id="11111111-1111-4111-8111-111111111111"):
    return TechnicalObservationEvent.model_validate(
        {
            "eventId": event_id,
            "schemaVersion": "1.0.0",
            "cameraExternalId": "CAM-GATE-01",
            "streamSessionId": "22222222-2222-4222-8222-222222222222",
            "capturedAt": "2026-09-27T10:00:00Z",
            "frameDimensions": {"width": 1280, "height": 720},
            "observations": [
                {
                    "type": "ZONE_ENTRY",
                    "trackId": 7,
                    "regionId": "33333333-3333-4333-8333-333333333333",
                    "geometryVersion": 1,
                    "confidence": 0.9,
                }
            ],
            "evidence": [],
        }
    )


@pytest.mark.anyio
async def test_original_context_survives_restart_without_changing_event(tmp_path: Path):
    path = tmp_path / "outbox.sqlite3"
    context = _context()
    event = _event()
    first = SqliteEventOutbox(path)
    assert await first.enqueue(event, delivery_context=context) == OutboxEnqueueOutcome.ENQUEUED
    (pending,) = await SqliteEventOutbox(path).ready()
    assert pending.delivery_context == context
    assert pending.event.to_wire_dict() == event.to_wire_dict()
    assert pending.payload_hash == event.compute_payload_hash()
    assert await first.enqueue(event, delivery_context=context) == OutboxEnqueueOutcome.IDEMPOTENT


@pytest.mark.anyio
async def test_context_cannot_be_relabelled_or_upgraded_from_legacy(tmp_path: Path):
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    event = _event()
    await outbox.enqueue(event, delivery_context=_context())
    for context in [None, _context(configurationVersion=3), _context(sourceKind="REPLAY")]:
        with pytest.raises(OutboxConflict):
            await outbox.enqueue(event, delivery_context=context)
    legacy = _event("44444444-4444-4444-8444-444444444444")
    await outbox.enqueue(legacy)
    with pytest.raises(OutboxConflict):
        await outbox.enqueue(legacy, delivery_context=_context())


@pytest.mark.anyio
async def test_bound_batch_is_atomic_and_context_matches_event_session(tmp_path: Path):
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    existing = _event()
    await outbox.enqueue(existing, delivery_context=_context())
    new = _event("44444444-4444-4444-8444-444444444444")
    with pytest.raises(OutboxConflict):
        await outbox.enqueue_many(
            (new, existing), delivery_context=_context(configurationVersion=3)
        )
    assert (await outbox.counts()).pending == 1
    with pytest.raises(ValueError):
        await outbox.enqueue(
            new, delivery_context=_context(streamSessionId="55555555-5555-4555-8555-555555555555")
        )


@pytest.mark.anyio
async def test_corrupted_or_partial_context_is_not_silently_legacy(tmp_path: Path):
    path = tmp_path / "outbox.sqlite3"
    outbox = SqliteEventOutbox(path)
    await outbox.enqueue(_event(), delivery_context=_context())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE observation_outbox SET delivery_context_json='{}'")
    with pytest.raises(RuntimeError, match="context"):
        await SqliteEventOutbox(path).ready()


@pytest.mark.anyio
async def test_bound_pending_event_cannot_fall_back_to_legacy_transport(tmp_path: Path):
    class LegacyClient:
        async def post_event(self, event):
            pytest.fail("Bound event was sent without its original execution context")

    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    await outbox.enqueue(_event(), delivery_context=_context())
    with pytest.raises(OutboxDeliveryBlocked, match="context"):
        await OutboxDispatcher(outbox, LegacyClient()).drain_once()
    assert (await outbox.counts()).pending == 1


def test_context_rejects_secret_extra_fields_wrong_versions_and_mutation():
    context = _context()
    with pytest.raises(ValueError):
        context.configuration_version = 4
    for changes in [
        {"secret": "do-not-store"},
        {"configurationVersion": True},
        {"workerGenerationId": "not-uuid"},
        {"sourceKind": "UNKNOWN"},
        {"receiptCommandId": None},
        {"viewRevision": 0},
    ]:
        with pytest.raises(ValueError):
            _context(**changes)


@pytest.mark.anyio
async def test_old_outbox_upgrade_preserves_pending_payload_and_retry_state(tmp_path: Path):
    import json

    path = tmp_path / "legacy.sqlite3"
    event = _event()
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE observation_outbox (
          event_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL,payload_hash TEXT NOT NULL,
          state TEXT NOT NULL,attempt_count INTEGER NOT NULL,next_attempt_at REAL NOT NULL,
          last_error_class TEXT,created_at REAL NOT NULL,updated_at REAL NOT NULL,delivered_at REAL
        ) STRICT""")
        payload = json.dumps(event.to_wire_dict())
        connection.execute(
            "INSERT INTO observation_outbox VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                event.event_id,
                payload,
                event.compute_payload_hash(),
                "PENDING",
                4,
                0,
                "SyntheticTimeout",
                1,
                2,
                None,
            ),
        )
    (pending,) = await SqliteEventOutbox(path).ready()
    assert pending.delivery_context is None
    assert pending.attempt_count == 4
    assert pending.payload_hash == event.compute_payload_hash()
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT payload_json,last_error_class,created_at,updated_at FROM observation_outbox"
        ).fetchone()
    assert row == (payload, "SyntheticTimeout", 1, 2)


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["partial", "changed-valid-context", "wrong-session"])
async def test_context_digest_and_pairing_fail_closed(tmp_path: Path, corruption: str):
    path = tmp_path / "context.sqlite3"
    outbox = SqliteEventOutbox(path)
    await outbox.enqueue(_event(), delivery_context=_context())
    with sqlite3.connect(path) as connection:
        if corruption == "partial":
            connection.execute("UPDATE observation_outbox SET delivery_context_hash=NULL")
        else:
            context = (
                _context(configurationVersion=3)
                if corruption == "changed-valid-context"
                else _context(streamSessionId="55555555-5555-4555-8555-555555555555")
            )
            connection.execute(
                "UPDATE observation_outbox SET delivery_context_json=?",
                (context.model_dump_json(by_alias=True),),
            )
    with pytest.raises(RuntimeError, match="context"):
        await SqliteEventOutbox(path).ready()


def test_four_process_initialization_does_not_race_additive_context_columns(tmp_path: Path):
    path = tmp_path / "concurrent.sqlite3"
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from smartsite_ai.integrations.outbox import SqliteEventOutbox; "
        "asyncio.run(SqliteEventOutbox(Path(sys.argv[1])).initialize()); print('READY')"
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stderr
            assert stdout.strip() == "READY"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(observation_outbox)")]
    assert columns.count("delivery_context_json") == 1
    assert columns.count("delivery_context_hash") == 1


def _credential(**changes):
    return WorkerGenerationCredential.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": _context().camera_id,
            "workerGenerationId": _context().worker_generation_id,
            "streamSessionId": _context().stream_session_id,
            "secret": "g" * 43,
            "expiresAt": "2026-09-27T11:00:00.000Z",
            **changes,
        }
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "base", ["http://backend.test", "http://backend.test/api/v1/integrations/ai/events"]
)
async def test_bound_sender_retries_original_envelope_and_credential(base):
    requests = []
    event, context = _event(), _context()

    def handle(request):
        requests.append(request)
        assert str(request.url) == "http://backend.test/api/v1/integrations/ai/worker-events"
        assert request.headers["authorization"] == "Bearer synthetic-service-token"
        assert request.headers["x-worker-generation-secret"] == "g" * 43
        assert json.loads(request.content) == {
            "schemaVersion": "1.0.0",
            "context": context.model_dump(mode="json", by_alias=True),
            "event": event.to_wire_dict(),
        }
        assert "g" * 43 not in request.content.decode()
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(
            202, json={"eventId": event.event_id, "status": "PROCESSED", "alertIds": []}
        )

    async with BackendClient(
        base,
        "synthetic-service-token",
        transport=httpx.MockTransport(handle),
        max_retries=1,
        backoff_factor=0,
    ) as client:
        assert callable(getattr(client, "post_bound_event", None))
        result = await client.post_bound_event(event, context, _credential())
    assert result["eventId"] == event.event_id
    assert len(requests) == 2
    assert requests[0].content == requests[1].content


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changes",
    [
        {"workerGenerationId": "dddddddd-dddd-4ddd-8ddd-dddddddddddd"},
        {"cameraId": "dddddddd-dddd-4ddd-8ddd-dddddddddddd"},
        {"streamSessionId": "dddddddd-dddd-4ddd-8ddd-dddddddddddd"},
    ],
)
async def test_bound_dispatch_does_not_replace_original_generation(tmp_path: Path, changes):
    def handle(request):
        pytest.fail("Mismatched generation credential reached HTTP")

    outbox = SqliteEventOutbox(tmp_path / "bound.sqlite3")
    await outbox.enqueue(_event(), delivery_context=_context())
    async with BackendClient(
        "http://backend.test", "synthetic-service-token", transport=httpx.MockTransport(handle)
    ) as client:
        dispatcher = OutboxDispatcher(outbox, client, generation_credential=_credential(**changes))
        with pytest.raises(OutboxDeliveryBlocked, match="credential"):
            await dispatcher.drain_once()
    assert (await outbox.counts()).pending == 1


@pytest.mark.anyio
async def test_bound_dispatch_marks_delivered_only_for_matching_response(tmp_path: Path):
    event = _event()
    seen = []

    def handle(request):
        seen.append(json.loads(request.content)["context"])
        return httpx.Response(
            202, json={"eventId": event.event_id, "status": "DUPLICATE_ACCEPTED", "alertIds": []}
        )

    outbox = SqliteEventOutbox(tmp_path / "bound.sqlite3")
    context = _context(sourceKind="REPLAY")
    await outbox.enqueue(event, delivery_context=context)
    async with BackendClient(
        "http://backend.test", "synthetic-service-token", transport=httpx.MockTransport(handle)
    ) as client:
        assert (
            await OutboxDispatcher(outbox, client, generation_credential=_credential()).drain_once()
            == 1
        )
    assert seen == [context.model_dump(mode="json", by_alias=True)]
    assert (await outbox.counts()).delivered == 1


@pytest.mark.anyio
async def test_bound_sender_does_not_follow_redirect_with_credentials():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://untrusted.test/collect"})

    async with BackendClient(
        "http://backend.test", "synthetic-service-token", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.post_bound_event(_event(), _context(), _credential())
    assert len(requests) == 1
