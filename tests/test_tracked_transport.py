import json
from pathlib import Path

import httpx
import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential

ROOT = Path(__file__).resolve().parents[1]


def packet():
    event = json.loads((ROOT / "contracts/examples/v1.7/one-person-camera-ppe.json").read_text())
    context = EventDeliveryContext.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": "11111111-1111-4111-8111-111111111111",
            "workerGenerationId": "22222222-2222-4222-8222-222222222222",
            "streamSessionId": event["streamSessionId"],
            "configurationVersion": 1,
            "bundleSha256": "a" * 64,
            "viewRevision": 1,
            "receiptCommandId": "33333333-3333-4333-8333-333333333333",
            "sourceKind": "REPLAY",
        }
    )
    credential = WorkerGenerationCredential.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": context.camera_id,
            "workerGenerationId": context.worker_generation_id,
            "streamSessionId": context.stream_session_id,
            "secret": "x" * 43,
            "expiresAt": "2030-01-01T00:00:00.000Z",
        }
    )
    envelope = {
        "schemaVersion": "1.0.0",
        "event": event,
        "context": context.model_dump(by_alias=True),
    }
    wire = json.dumps(envelope, indent=3).encode()
    digest = compute_canonical_payload_hash(event)
    receipt = {
        "kind": "RAW_STORED",
        "eventId": event["eventId"].lower(),
        "payloadHash": digest,
        "bindingHash": context.binding_hash(digest),
    }
    return event, context, credential, envelope, wire, receipt


@pytest.mark.anyio
async def test_exact_retained_bytes_receive_only_bound_raw_receipt():
    event, context, credential, _, wire, receipt = packet()
    requests = []

    def backend(request):
        requests.append(request)
        assert request.url.path == "/api/v1/integrations/ai/tracked-parts"
        assert request.content == wire
        assert request.headers["x-worker-generation-secret"] == credential.secret.get_secret_value()
        return httpx.Response(200, json=receipt)

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        result = await client.post_tracked_part(credential, wire)
    assert result.kind == "RAW_STORED"
    assert result.event_id == event["eventId"].lower()
    assert result.binding_hash == context.binding_hash(receipt["payloadHash"])
    assert len(requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", ["orphan", "duplicate-json", "wrong-session", "oversized", "unknown-version"]
)
async def test_bad_original_envelope_is_rejected_before_any_network(case):
    _, _, credential, envelope, wire, _ = packet()
    if case == "orphan":
        envelope["event"]["observations"][1]["trackId"] = 500
    elif case == "duplicate-json":
        wire = wire.replace(
            b'"schemaVersion": "1.0.0"', b'"schemaVersion": "1.0.0", "schemaVersion": "1.0.0"', 1
        )
    elif case == "wrong-session":
        envelope["event"]["streamSessionId"] = "44444444-4444-4444-8444-444444444444"
    elif case == "oversized":
        wire = b" " * (1024 * 1024 + 1)
    else:
        envelope["event"]["schemaVersion"] = "1.8.0"
    if case not in {"duplicate-json", "oversized"}:
        wire = json.dumps(envelope).encode()

    def backend(_):
        pytest.fail("Invalid original bytes must not reach the network")

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        with pytest.raises((ValueError, RuntimeError)):
            await client.post_tracked_part(credential, wire)


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["wrong-hash", "wrong-context", "duplicate-json", "oversized"])
async def test_raw_receipt_cannot_advance_on_mismatch_or_ambiguous_response(case):
    _, _, credential, _, wire, receipt = packet()
    if case == "wrong-hash":
        receipt["payloadHash"] = "0" * 64
    elif case == "wrong-context":
        receipt["bindingHash"] = "0" * 64
    body = json.dumps(receipt).encode()
    if case == "duplicate-json":
        body = body.replace(b'"kind": "RAW_STORED"', b'"kind": "RAW_STORED", "kind": "RAW_STORED"')
    elif case == "oversized":
        body = b" " * 4097
    async with BackendClient(
        "https://backend.example",
        "synthetic-service",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content=body, headers={"content-type": "application/json"}
            )
        ),
    ) as client:
        with pytest.raises(ValueError):
            await client.post_tracked_part(credential, wire)


@pytest.mark.anyio
async def test_tracked_model_is_never_sent_to_legacy_ingestion_routes():
    event, context, credential, _, _, _ = packet()
    tracked = TrackedObservationEvent.model_validate(event)

    def backend(_):
        pytest.fail("Tracked version must not reach a legacy ingestion route")

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        with pytest.raises(ValueError):
            await client.post_event(tracked)
        with pytest.raises(ValueError):
            await client.post_bound_event(tracked, context, credential)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    ["valid", "pending", "raw-only", "wrong-scope", "wrong-person", "duplicate-json", "oversized"],
)
async def test_only_exact_admitted_seal_selects_the_expected_original_subject(case):
    event, _, credential, _, _, raw_receipt = packet()
    segment = event["continuity"]["subjects"][0]["segmentId"]
    expected = {segment: (event["eventId"].lower(), 0)}
    result = {
        "kind": "ADMITTED_TRACKED_FRAME",
        "scopeSha256": "a" * 64,
        "sequenceNumber": event["framePartition"]["sequenceNumber"],
        "subjects": [
            {"segmentId": segment, "eventId": event["eventId"].lower(), "personObservationIndex": 0}
        ],
        "retiredSegmentIds": [],
    }
    if case == "wrong-scope":
        result["scopeSha256"] = "0" * 64
    elif case == "wrong-person":
        result["subjects"][0]["personObservationIndex"] = 1
    body = json.dumps(raw_receipt if case == "raw-only" else result).encode()
    if case == "duplicate-json":
        body = body.replace(
            b'"kind": "ADMITTED_TRACKED_FRAME"',
            b'"kind": "ADMITTED_TRACKED_FRAME", "kind": "ADMITTED_TRACKED_FRAME"',
        )
    elif case == "oversized":
        body = b" " * (1024 * 1024 + 1)

    def backend(request):
        assert request.method == "GET"
        assert str(request.url).endswith(
            "/tracked-frames/" + event["framePartition"]["frameId"] + "/result"
        )
        return httpx.Response(
            404 if case == "pending" else 200,
            content=body,
            headers={"content-type": "application/json"},
        )

    async with BackendClient(
        "https://backend.example", "synthetic-service", transport=httpx.MockTransport(backend)
    ) as client:
        read = client.read_tracked_frame_result(
            credential,
            event["framePartition"]["frameId"],
            scope_sha256="a" * 64,
            sequence_number=result["sequenceNumber"],
            subjects=expected,
            retired_segment_ids=set(),
        )
        if case in {"valid", "pending"}:
            admitted = await read
            if case == "pending":
                assert admitted is None
            else:
                assert admitted.subject_references() == expected
        else:
            with pytest.raises(ValueError):
                await read
