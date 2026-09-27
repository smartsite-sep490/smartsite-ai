from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def _event(
    event_id: str = "11111111-1111-4111-8111-111111111111",
    *,
    camera_external_id: str = "CAM-GATE-01",
) -> TechnicalObservationEvent:
    return TechnicalObservationEvent.model_validate(
        {
            "eventId": event_id,
            "schemaVersion": "1.0.0",
            "cameraExternalId": camera_external_id,
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
async def test_outbox_persists_pending_event_across_instances(tmp_path: Path) -> None:
    clock = MutableClock()
    path = (tmp_path / "events.sqlite3").resolve()
    first = SqliteEventOutbox(path, clock=clock)

    assert await first.enqueue(_event()) is OutboxEnqueueOutcome.ENQUEUED

    reopened = SqliteEventOutbox(path, clock=clock)
    pending = await reopened.ready()
    assert len(pending) == 1
    assert pending[0].event == _event()
    assert pending[0].payload_hash == _event().compute_payload_hash()
    assert pending[0].attempt_count == 0
    assert (await reopened.counts()).pending == 1


@pytest.mark.anyio
async def test_enqueue_is_idempotent_and_rejects_same_id_with_different_content(
    tmp_path: Path,
) -> None:
    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve())
    original = _event()

    assert await outbox.enqueue(original) is OutboxEnqueueOutcome.ENQUEUED
    assert await outbox.enqueue(original) is OutboxEnqueueOutcome.IDEMPOTENT

    with pytest.raises(OutboxConflict, match="different canonical content"):
        await outbox.enqueue(_event(camera_external_id="CAM-YARD-02"))
    assert (await outbox.counts()).pending == 1


@pytest.mark.anyio
async def test_outbox_validates_path_and_ready_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        SqliteEventOutbox(Path("relative.sqlite3"))
    with pytest.raises(ValueError, match="parent directory"):
        SqliteEventOutbox((tmp_path / "missing" / "events.sqlite3").resolve())

    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve())
    with pytest.raises(ValueError, match="between 1 and 1024"):
        await outbox.ready(limit=0)


@pytest.mark.anyio
async def test_dispatch_success_marks_delivered_and_sends_exact_payload(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    event = _event()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            202,
            json={"eventId": event.event_id, "status": "PROCESSED", "alertIds": []},
        )

    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve())
    await outbox.enqueue(event)
    async with BackendClient(
        "http://backend:3000",
        "secret-token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        dispatcher = OutboxDispatcher(outbox, backend)
        assert await dispatcher.drain_once() == 1

    assert len(requests) == 1
    assert json.loads(requests[0].content) == event.to_wire_dict()
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    counts = await outbox.counts()
    assert counts.delivered == 1
    assert counts.pending == 0
    assert await outbox.ready() == ()


@pytest.mark.anyio
async def test_dispatch_nonretryable_4xx_marks_terminal_without_payload_leak(
    tmp_path: Path,
) -> None:
    event = _event()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(422, json={"error": "invalid"})

    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve())
    await outbox.enqueue(event)
    async with BackendClient(
        "http://backend:3000",
        "secret-token",
        transport=httpx.MockTransport(handler),
    ) as backend:
        dispatcher = OutboxDispatcher(outbox, backend)
        assert await dispatcher.drain_once() == 1

    assert calls == 1
    counts = await outbox.counts()
    assert counts.terminal == 1
    assert counts.pending == 0


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_operator_fixable_4xx_preserves_event_and_blocks_worker(
    tmp_path: Path,
    status: int,
) -> None:
    clock = MutableClock()
    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve(), clock=clock)
    await outbox.enqueue(_event())
    async with BackendClient(
        "http://backend:3000",
        "expired-token",
        transport=httpx.MockTransport(lambda _: httpx.Response(status)),
        max_retries=0,
    ) as backend:
        dispatcher = OutboxDispatcher(outbox, backend)
        with pytest.raises(OutboxDeliveryBlocked, match=f"HTTP {status}"):
            await dispatcher.drain_once()

    counts = await outbox.counts()
    assert counts.pending == 1
    assert counts.terminal == 0
    assert await outbox.ready() == ()
    clock.advance(1)
    assert len(await outbox.ready()) == 1


@pytest.mark.anyio
async def test_transient_failure_is_rescheduled_and_retried_after_due_time(tmp_path: Path) -> None:
    clock = MutableClock()
    event = _event()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("backend unavailable", request=request)
        return httpx.Response(
            200,
            json={"eventId": event.event_id, "status": "DUPLICATE_ACCEPTED", "alertIds": []},
        )

    async def no_sleep(_delay: float) -> None:
        return None

    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve(), clock=clock)
    await outbox.enqueue(event)
    async with BackendClient(
        "http://backend:3000",
        "secret-token",
        transport=httpx.MockTransport(handler),
        sleep_func=no_sleep,
        max_retries=0,
    ) as backend:
        dispatcher = OutboxDispatcher(
            outbox,
            backend,
            retry_base=timedelta(seconds=2),
            retry_max=timedelta(seconds=8),
        )
        assert await dispatcher.drain_once() == 1
        assert await outbox.ready() == ()
        clock.advance(2)
        ready = await outbox.ready()
        assert len(ready) == 1
        assert ready[0].attempt_count == 1
        assert await dispatcher.drain_once() == 1

    assert calls == 2
    assert (await outbox.counts()).delivered == 1


@pytest.mark.anyio
async def test_malformed_success_ack_is_retried_with_same_event_id(tmp_path: Path) -> None:
    clock = MutableClock()
    event = _event()
    submitted_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        submitted_ids.append(json.loads(request.content)["eventId"])
        if len(submitted_ids) == 1:
            return httpx.Response(202, json={"unexpected": True})
        return httpx.Response(
            200,
            json={"eventId": event.event_id, "status": "DUPLICATE_ACCEPTED", "alertIds": []},
        )

    outbox = SqliteEventOutbox((tmp_path / "events.sqlite3").resolve(), clock=clock)
    await outbox.enqueue(event)
    async with BackendClient(
        "http://backend:3000",
        "secret-token",
        transport=httpx.MockTransport(handler),
        max_retries=0,
    ) as backend:
        dispatcher = OutboxDispatcher(outbox, backend)
        assert await dispatcher.drain_once() == 1
        clock.advance(1)
        assert await dispatcher.drain_once() == 1

    assert submitted_ids == [event.event_id, event.event_id]
    assert (await outbox.counts()).delivered == 1


@pytest.mark.anyio
async def test_outbox_detects_persisted_payload_corruption(tmp_path: Path) -> None:
    import sqlite3

    path = (tmp_path / "events.sqlite3").resolve()
    outbox = SqliteEventOutbox(path)
    await outbox.enqueue(_event())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE observation_outbox SET payload_json = ?",
            ('{"eventId":"corrupt"}',),
        )

    reopened = SqliteEventOutbox(path)
    with pytest.raises(RuntimeError, match="invalid observation event"):
        await reopened.ready()
