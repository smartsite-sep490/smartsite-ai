import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_tracked_transport import packet

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent
from smartsite_ai.domain.tracked_raw_receipt import TrackedRawReceipt
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import (
    OutboxConflict,
    OutboxDispatcher,
    SqliteEventOutbox,
    TrackedOutboxCapacityReached,
)

NOW = datetime(2026, 11, 17, tzinfo=UTC)


def frame_parts():
    event, context, credential, *_ = packet()
    first = copy.deepcopy(event)
    first["framePartition"]["partCount"] = 2
    second = copy.deepcopy(first)
    second["eventId"] = "00000000-0000-4000-8000-000000000099"
    second["framePartition"]["partIndex"] = 1
    second["observations"][0]["trackId"] = 1
    second["observations"][1]["trackId"] = 1
    second["continuity"]["subjects"][0]["segmentId"] = "00000000-0000-4000-8000-000000000098"
    refs = TrackedFrameReferences.model_validate(
        {
            "scopeSha256": "b" * 64,
            "sequenceNumber": first["framePartition"]["sequenceNumber"],
            "subjects": [
                {
                    "segmentId": item["continuity"]["subjects"][0]["segmentId"],
                    "eventId": item["eventId"],
                    "personObservationIndex": 0,
                }
                for item in (first, second)
            ],
            "retiredSegmentIds": [],
        }
    )
    return (first, second), context, credential, refs


async def enqueue(outbox, parts=None, refs=None, **options):
    defaults, context, credential, expected = frame_parts()
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        defaults if parts is None else parts,
        expected_seal=expected if refs is None else refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=2,
        max_bytes=options.get("max_bytes", 100_000),
    )


@pytest.mark.anyio
async def test_delivery_exposes_an_immediately_admitted_frame_before_next_tick(tmp_path):
    _, context, credential, refs = frame_parts()
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    await enqueue(outbox)
    methods = []

    def backend(request):
        methods.append(request.method)
        if request.method == "POST":
            event = json.loads(request.content)["event"]
            digest = compute_canonical_payload_hash(event)
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"],
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        return httpx.Response(
            200, json={"kind": "ADMITTED_TRACKED_FRAME", **refs.model_dump(by_alias=True)}
        )

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        dispatcher = OutboxDispatcher(
            outbox, client, generation_credential=credential, tracked_batch_limit=2
        )
        await dispatcher.drain_once()
        assert methods == ["POST", "POST", "GET"]
        head = await outbox.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number)
        assert head is not None
        assert head.subject_references() == refs.subject_references()
        assert not await outbox.has_pending_tracked_frame(credential)

        await dispatcher.drain_once()
        assert methods == ["POST", "POST", "GET"]


@pytest.mark.anyio
async def test_tracked_outbox_restart_atomic_retry_and_admitted_head(tmp_path):
    parts, context, credential, refs = frame_parts()
    path = tmp_path / "outbox.sqlite"
    outbox = SqliteEventOutbox(path, clock=lambda: NOW)
    await enqueue(outbox)
    await enqueue(outbox)
    reopened = SqliteEventOutbox(path, clock=lambda: NOW)
    ((frame_id, wires),) = await reopened.ready_tracked_parts(credential, limit=2)
    assert [json.loads(w)["event"] for w in wires] == list(parts)
    assert all(credential.secret.get_secret_value().encode() not in w for w in wires)
    assert not await reopened.ready()
    assert await reopened.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number) is None
    other = credential.model_copy(update={"stream_session_id": parts[1]["eventId"]})
    assert not await reopened.ready_tracked_parts(other, limit=2)
    posted = []
    fail_second = True
    pending = True

    def backend(request):
        nonlocal fail_second
        if request.method == "POST":
            posted.append(request.content)
            if fail_second and request.content == wires[1]:
                fail_second = False
                return httpx.Response(503)
            event = json.loads(request.content)["event"]
            digest = compute_canonical_payload_hash(event)
            return httpx.Response(
                200,
                json={
                    "kind": "RAW_STORED",
                    "eventId": event["eventId"],
                    "payloadHash": digest,
                    "bindingHash": context.binding_hash(digest),
                },
            )
        assert request.method == "GET"
        if pending:
            return httpx.Response(404)
        return httpx.Response(
            200, json={"kind": "ADMITTED_TRACKED_FRAME", **refs.model_dump(by_alias=True)}
        )

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        dispatcher = OutboxDispatcher(reopened, client, generation_credential=credential)
        with pytest.raises(httpx.HTTPStatusError):
            await dispatcher.drain_tracked_once(limit=2)
        assert await reopened.ready_tracked_parts(credential, limit=2) == ((frame_id, wires),)
        assert await dispatcher.drain_tracked_once(limit=2) == 1
        assert posted == [*wires, *wires]
        assert not await reopened.ready_tracked_parts(credential, limit=2)
        assert (
            await reopened.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number) is None
        )
        assert await reopened.reconcile_tracked_frames(client, credential, limit=2) == 0
        pending = False
        assert await reopened.reconcile_tracked_frames(client, credential, limit=2) == 1
    final = SqliteEventOutbox(path, clock=lambda: NOW)
    head = await final.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number)
    assert head.subject_references() == refs.subject_references()
    assert not await final.has_pending_tracked_frame(credential)
    assert await final.read_tracked_predecessor(refs.scope_sha256, "99") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "orphan",
        "different-profile",
        "different-captured-at",
        "different-dimensions",
        "different-camera",
        "duplicate-track",
        "duplicate-segment",
        "duplicate-retirement",
        "active-retirement",
        "duplicate-predecessor",
        "unknown-version",
        "wrong-session",
        "size",
    ],
)
async def test_invalid_parts_never_leave_partial_durable_work(tmp_path, case):
    parts, _, credential, _ = frame_parts()
    first, second = parts
    if case == "orphan":
        second["observations"][1]["trackId"] = 99
    elif case == "different-profile":
        second["continuity"]["profileSha256"] = "c" * 64
    elif case == "different-captured-at":
        second["capturedAt"] = "2026-11-01T00:00:01.000Z"
    elif case == "different-dimensions":
        second["frameDimensions"]["width"] = 640
    elif case == "different-camera":
        second["cameraExternalId"] = "another-camera"
    elif case == "duplicate-track":
        for item in second["observations"]:
            item["trackId"] = 0
    elif case == "duplicate-segment":
        second["continuity"]["subjects"][0]["segmentId"] = first["continuity"]["subjects"][0][
            "segmentId"
        ]
    elif case in {"duplicate-retirement", "active-retirement"}:
        retirement = {"segmentId": "00000000-0000-4000-8000-000000000088", "reason": "LOST_TRACK"}
        first["continuity"]["retiredSegments"] = [retirement]
        if case == "duplicate-retirement":
            second["continuity"]["retiredSegments"] = [{**retirement, "reason": "RECONNECT"}]
        else:
            retirement["segmentId"] = second["continuity"]["subjects"][0]["segmentId"]
    elif case == "duplicate-predecessor":
        for item in parts:
            item["continuity"]["subjects"][0]["link"] = {
                "kind": "CONTINUE",
                "previousEventId": "00000000-0000-4000-8000-000000000077",
                "previousPersonObservationIndex": 0,
            }
    elif case == "unknown-version":
        second["schemaVersion"] = "1.8.0"
    elif case == "wrong-session":
        second["streamSessionId"] = "00000000-0000-4000-8000-000000000077"
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    with pytest.raises(ValueError):
        await enqueue(outbox, parts, max_bytes=1 if case == "size" else 100_000)
    assert not await outbox.ready_tracked_parts(credential, limit=2)
    assert not await outbox.has_pending_tracked_frame(credential)


@pytest.mark.anyio
async def test_same_frame_changed_bytes_conflict_without_overwriting(tmp_path):
    parts, _, credential, _ = frame_parts()
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    await enqueue(outbox)
    before = await outbox.ready_tracked_parts(credential, limit=2)
    parts[1]["observations"][1]["status"] = "PRESENT"
    with pytest.raises(OutboxConflict):
        await enqueue(outbox, parts)
    assert await outbox.ready_tracked_parts(credential, limit=2) == before


@pytest.mark.anyio
async def test_receipt_binding_and_duplicate_receipts_never_promote_raw(tmp_path):
    _, context, credential, refs = frame_parts()
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    await enqueue(outbox)
    ((frame_id, wires),) = await outbox.ready_tracked_parts(credential, limit=2)
    receipts = []
    for wire in wires:
        event = json.loads(wire)["event"]
        digest = compute_canonical_payload_hash(event)
        receipts.append(
            TrackedRawReceipt(
                kind="RAW_STORED",
                eventId=event["eventId"],
                payloadHash=digest,
                bindingHash=context.binding_hash(digest),
            )
        )
    wrong = receipts[1].model_copy(update={"binding_hash": "f" * 64})
    for invalid in ((receipts[0],), (receipts[0], receipts[0]), (receipts[0], wrong)):
        with pytest.raises(ValueError):
            await outbox.confirm_tracked_parts_stored(credential, frame_id, invalid)
        assert await outbox.ready_tracked_parts(credential, limit=2) == ((frame_id, wires),)
        assert (
            await outbox.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number) is None
        )


@pytest.mark.anyio
async def test_expired_frame_remains_unresolved_without_becoming_a_predecessor(tmp_path):
    _, _, credential, refs = frame_parts()
    now = NOW
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: now)
    await enqueue(outbox)
    assert await outbox.tracked_pending_remaining_seconds(credential) == 30
    now += timedelta(seconds=31)
    assert not await outbox.ready_tracked_parts(credential, limit=2)
    assert await outbox.has_pending_tracked_frame(credential)
    assert await outbox.tracked_pending_remaining_seconds(credential) == -1
    assert await outbox.read_tracked_predecessor(refs.scope_sha256, refs.sequence_number) is None


@pytest.mark.anyio
async def test_capacity_is_atomic_for_competing_connections(tmp_path):
    import asyncio

    parts, context, credential, refs = frame_parts()
    path = tmp_path / "outbox.sqlite"
    left = SqliteEventOutbox(path, clock=lambda: NOW)
    right = SqliteEventOutbox(path, clock=lambda: NOW)
    other = copy.deepcopy(parts)
    for part in other:
        part["framePartition"]["frameId"] = "00000000-0000-4000-8000-000000000066"

    async def stage(box, frame):
        await box.enqueue_tracked_parts(
            credential,
            context,
            frame,
            expected_seal=refs,
            expires_at=NOW + timedelta(seconds=30),
            max_pending=1,
            max_bytes=100_000,
        )

    outcomes = await asyncio.gather(stage(left, parts), stage(right, other), return_exceptions=True)
    assert sum(item is None for item in outcomes) == 1
    assert sum(isinstance(item, TrackedOutboxCapacityReached) for item in outcomes) == 1
    assert len(await left.ready_tracked_parts(credential, limit=2)) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("many", [False, True])
async def test_tracked_subtype_cannot_enter_the_legacy_outbox(tmp_path, many):
    event, *_ = packet()
    tracked = TrackedObservationEvent.model_validate(event)
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    with pytest.raises(ValueError):
        if many:
            await outbox.enqueue_many((tracked,))
        else:
            await outbox.enqueue(tracked)
    assert not await outbox.ready()


@pytest.mark.anyio
async def test_large_subject_repeated_context_uses_first_original_person(tmp_path):
    parts, _, credential, refs = frame_parts()
    first, second = parts
    # Canonical packing repeats PERSON for disjoint region groups of a large subject.
    second["observations"] = copy.deepcopy(first["observations"][:1])
    second["continuity"]["subjects"] = copy.deepcopy(first["continuity"]["subjects"])
    refs = TrackedFrameReferences.model_validate(
        {
            **refs.model_dump(by_alias=True),
            "subjects": [refs.model_dump(by_alias=True)["subjects"][0]],
        }
    )
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite", clock=lambda: NOW)
    await enqueue(outbox, parts, refs)
    assert len(await outbox.ready_tracked_parts(credential, limit=2)) == 1


@pytest.mark.anyio
async def test_uppercase_original_ids_keep_bytes_and_match_lowercase_backend_receipts(tmp_path):
    event, context, credential, *_ = packet()
    event["eventId"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    event["framePartition"]["frameId"] = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    session = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    context = context.model_copy(update={"stream_session_id": session})
    credential = credential.model_copy(update={"stream_session_id": session})
    event["streamSessionId"] = session.upper()
    refs = TrackedFrameReferences.model_validate(
        {
            "scopeSha256": "a" * 64,
            "sequenceNumber": "0",
            "retiredSegmentIds": [],
            "subjects": [
                {
                    "segmentId": event["continuity"]["subjects"][0]["segmentId"],
                    "eventId": event["eventId"].lower(),
                    "personObservationIndex": 0,
                }
            ],
        }
    )
    outbox = SqliteEventOutbox(tmp_path / "uppercase.sqlite", clock=lambda: NOW)
    await outbox.enqueue_tracked_parts(
        credential,
        context,
        (event,),
        expected_seal=refs,
        expires_at=NOW + timedelta(seconds=30),
        max_pending=1,
        max_bytes=100_000,
    )
    ((frame_id, (wire,)),) = await outbox.ready_tracked_parts(credential, limit=1)
    assert frame_id == event["framePartition"]["frameId"].lower()
    assert json.loads(wire)["event"] == event

    def backend(request):
        assert request.content == wire
        digest = compute_canonical_payload_hash(event)
        return httpx.Response(
            200,
            json={
                "kind": "RAW_STORED",
                "eventId": event["eventId"].lower(),
                "payloadHash": digest,
                "bindingHash": context.binding_hash(digest),
            },
        )

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        dispatcher = OutboxDispatcher(outbox, client, generation_credential=credential)
        assert await dispatcher.drain_tracked_once(limit=1) == 1
    assert not await outbox.ready_tracked_parts(credential, limit=1)
    assert await outbox.read_tracked_predecessor(refs.scope_sha256, "0") is None


@pytest.mark.anyio
async def test_case_variant_part_ids_are_one_identity_not_two_durable_events(tmp_path):
    parts, context, credential, _ = frame_parts()
    for part, event_id in zip(
        parts,
        ("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        strict=True,
    ):
        part["eventId"] = event_id
        part["observations"] = []
        part["continuity"]["subjects"] = []
        part["continuity"]["coverage"] = {"status": "INCOMPLETE", "reason": "INFERENCE_ERROR"}
    outbox = SqliteEventOutbox(tmp_path / "duplicate.sqlite", clock=lambda: NOW)
    with pytest.raises(ValueError, match="original frame"):
        await outbox.enqueue_tracked_parts(
            credential,
            context,
            parts,
            expected_seal=None,
            expires_at=NOW + timedelta(seconds=30),
            max_pending=1,
            max_bytes=100_000,
        )
    assert not await outbox.has_pending_tracked_frame(credential)
