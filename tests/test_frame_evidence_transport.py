import copy
import gzip
import hashlib
import io
import json
from email.parser import BytesParser
from email.policy import default as email_policy

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr
from test_tracked_transport import packet

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.integrations.backend_client import (
    BackendClient,
    BoundDeliveryCredentialUnavailable,
)

SERVICE_TOKEN = "secret-service-token-value-987654"
GENERATION_SECRET = "secret-generation-value-" + ("g" * 24)


def make_jpeg_bytes(
    *, width: int = 32, height: int = 24, color: tuple[int, int, int] = (40, 90, 150)
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    assert raw.startswith(b"\xff\xd8")
    assert raw.endswith(b"\xff\xd9")
    return raw


def live_evidence_packet(
    *,
    source_kind: str = "LIVE",
    with_frame: bool = True,
    image: bytes | None = None,
):
    event, context, credential, envelope, _, _ = packet()
    event = copy.deepcopy(event)
    event["frameDimensions"] = {"width": 32, "height": 24}
    context = EventDeliveryContext.model_validate(
        {
            **context.model_dump(mode="json", by_alias=True),
            "sourceKind": source_kind,
        }
    )
    credential = credential.model_copy(update={"secret": SecretStr(GENERATION_SECRET)})
    jpeg = make_jpeg_bytes() if image is None else image
    digest = hashlib.sha256(jpeg).hexdigest()
    session = event["streamSessionId"].lower()
    sequence = event["framePartition"]["sequenceNumber"]
    event_id = event["eventId"].lower()
    if with_frame:
        event["evidence"] = [
            {
                "kind": "FRAME",
                "uri": f"local://evidence/{session}/{sequence}/{event_id}.jpg",
                "sha256": digest,
            }
        ]
    else:
        event["evidence"] = []

    envelope = {
        "schemaVersion": "1.0.0",
        "event": event,
        "context": context.model_dump(mode="json", by_alias=True),
    }
    wire = json.dumps(envelope, indent=2).encode("utf-8")
    payload_hash = compute_canonical_payload_hash(event)
    receipt = {
        "schemaVersion": "1.0.0",
        "kind": "EVIDENCE_STORED",
        "eventId": event_id,
        "payloadHash": payload_hash,
        "bindingHash": context.binding_hash(payload_hash),
        "sha256": digest,
    }
    return event, context, credential, envelope, wire, jpeg, receipt


def parse_multipart(request: httpx.Request) -> dict[str, tuple[str | None, str, bytes]]:
    content_type = request.headers["content-type"]
    raw_message = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        + request.content
    )
    parsed = BytesParser(policy=email_policy).parsebytes(raw_message)
    assert parsed.is_multipart()
    parts: dict[str, tuple[str | None, str, bytes]] = {}
    for part in parsed.iter_parts():
        name = part.get_param("name", header="content-disposition")
        assert isinstance(name, str) and name not in parts
        filename = part.get_filename()
        parts[name] = (filename, part.get_content_type(), part.get_payload(decode=True))
    return parts


def _assert_no_secrets_in_error(exc: BaseException) -> None:
    text = f"{exc} | {exc!r}"
    assert SERVICE_TOKEN not in text
    assert GENERATION_SECRET not in text


@pytest.mark.anyio
async def test_upload_tracked_frame_sends_exact_multipart_and_validates_receipt():
    event, context, credential, envelope, wire, jpeg, receipt = live_evidence_packet()
    requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/v1/integrations/ai/tracked-parts/evidence"
        assert request.url.query == b""
        assert request.headers["authorization"] == f"Bearer {SERVICE_TOKEN}"
        assert request.headers["x-worker-generation-secret"] == GENERATION_SECRET
        assert request.headers["accept-encoding"] == "identity"

        parts = parse_multipart(request)
        assert set(parts) == {"schemaVersion", "event", "context", "frame"}
        assert parts["schemaVersion"][2] == b"1.0.0"
        assert json.loads(parts["event"][2].decode("utf-8")) == event
        assert json.loads(parts["context"][2].decode("utf-8")) == envelope["context"]
        filename, mime, body = parts["frame"]
        assert filename == "frame.jpg"
        assert mime == "image/jpeg"
        assert body == jpeg
        return httpx.Response(200, json=receipt)

    async with BackendClient(
        "https://backend.example/api/v1/integrations/ai/events",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        stored = await client.upload_tracked_frame(credential, wire, jpeg)

    assert len(requests) == 1
    assert stored.kind == "EVIDENCE_STORED"
    assert stored.event_id == receipt["eventId"]
    assert stored.payload_hash == receipt["payloadHash"]
    assert stored.binding_hash == receipt["bindingHash"]
    assert stored.sha256 == hashlib.sha256(jpeg).hexdigest()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "replay-context",
        "no-frame",
        "wrong-jpeg-magic",
        "truncated-jpeg-eoi",
        "tiny-image",
        "oversized-image",
        "wrong-hash",
        "duplicate-json-wire",
        "oversized-wire",
        "extra-envelope-key",
        "wrong-camera-credential",
        "wrong-generation-credential",
        "wrong-session-credential",
        "event-session-mismatch",
    ],
)
async def test_upload_tracked_frame_rejects_invalid_inputs_before_network(case):
    event, context, credential, envelope, wire, jpeg, _ = live_evidence_packet()

    if case == "replay-context":
        _, _, _, _, wire, jpeg, _ = live_evidence_packet(source_kind="REPLAY")
    elif case == "no-frame":
        _, _, _, _, wire, jpeg, _ = live_evidence_packet(with_frame=False)
    elif case == "wrong-jpeg-magic":
        jpeg = b"\x89PNG" + jpeg[4:]
    elif case == "truncated-jpeg-eoi":
        jpeg = jpeg[:-2] + b"\x00\x00"
    elif case == "tiny-image":
        jpeg = b"\xff\xd8\xff"
    elif case == "oversized-image":
        jpeg = b"\xff\xd8" + (b"\x00" * (1_048_577 - 4)) + b"\xff\xd9"
    elif case == "wrong-hash":
        jpeg = make_jpeg_bytes(color=(250, 10, 10))
    elif case == "duplicate-json-wire":
        wire = wire.replace(
            b'"schemaVersion": "1.0.0"',
            b'"schemaVersion": "1.0.0", "schemaVersion": "1.0.0"',
            1,
        )
    elif case == "oversized-wire":
        wire = b" " * (1024 * 1024 + 1)
    elif case == "extra-envelope-key":
        wire = json.dumps({**envelope, "extra": True}).encode("utf-8")
    elif case == "wrong-camera-credential":
        credential = credential.model_copy(
            update={"camera_id": "99999999-9999-4999-8999-999999999999"}
        )
    elif case == "wrong-generation-credential":
        credential = credential.model_copy(
            update={"worker_generation_id": "99999999-9999-4999-8999-999999999999"}
        )
    elif case == "wrong-session-credential":
        credential = credential.model_copy(
            update={"stream_session_id": "99999999-9999-4999-8999-999999999999"}
        )
    elif case == "event-session-mismatch":
        envelope["event"]["streamSessionId"] = "99999999-9999-4999-8999-999999999999"
        wire = json.dumps(envelope).encode("utf-8")

    def fail_network(_: httpx.Request) -> httpx.Response:
        pytest.fail(f"Case {case} must fail before reaching the network")

    expected_exc = (
        BoundDeliveryCredentialUnavailable
        if case
        in {
            "wrong-camera-credential",
            "wrong-generation-credential",
            "wrong-session-credential",
            "event-session-mismatch",
        }
        else ValueError
    )
    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(fail_network),
    ) as client:
        with pytest.raises(expected_exc) as exc_info:
            await client.upload_tracked_frame(credential, wire, jpeg)
    _assert_no_secrets_in_error(exc_info.value)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case", "status_code", "extra_headers"),
    [
        ("tampered-kind", 200, {}),
        ("tampered-event-id", 200, {}),
        ("tampered-payload-hash", 200, {}),
        ("tampered-binding-hash", 200, {}),
        ("tampered-sha256", 200, {}),
        ("extra-field", 200, {}),
        ("duplicate-json", 200, {}),
        ("oversized-body", 200, {}),
        ("gzip-encoded", 200, {"content-encoding": "gzip"}),
        ("wrong-content-type", 200, {"content-type": "text/plain"}),
        ("non-200-2xx", 201, {}),
        ("redirect-302", 302, {"location": "https://evil.example/steal"}),
        ("http-400", 400, {}),
        ("http-401", 401, {}),
        ("http-403", 403, {}),
        ("http-408", 408, {}),
        ("http-409", 409, {}),
        ("http-413", 413, {}),
        ("http-429", 429, {}),
        ("http-500", 500, {}),
        ("http-503", 503, {}),
    ],
)
async def test_upload_tracked_frame_rejects_bad_responses_without_retrying_or_leaking_secrets(
    case, status_code, extra_headers
):
    _, _, credential, _, wire, jpeg, receipt = live_evidence_packet()
    receipt = dict(receipt)

    if case == "tampered-kind":
        receipt["kind"] = "RAW_STORED"
    elif case == "tampered-event-id":
        receipt["eventId"] = "99999999-9999-4999-8999-999999999999"
    elif case == "tampered-payload-hash":
        receipt["payloadHash"] = "0" * 64
    elif case == "tampered-binding-hash":
        receipt["bindingHash"] = "0" * 64
    elif case == "tampered-sha256":
        receipt["sha256"] = "0" * 64
    elif case == "extra-field":
        receipt["publicUrl"] = "https://backend.example/public.jpg"

    body = json.dumps(receipt).encode("utf-8")
    if case == "duplicate-json":
        body = body.replace(
            b'"kind": "EVIDENCE_STORED"',
            b'"kind": "EVIDENCE_STORED", "kind": "EVIDENCE_STORED"',
            1,
        )
    elif case == "oversized-body":
        body = b"{" + (b" " * 4097) + b"}"
    elif case == "gzip-encoded":
        body = gzip.compress(body)

    attempts = 0

    def backend(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        headers = {"content-type": "application/json", **extra_headers}
        return httpx.Response(status_code, content=body, headers=headers)

    expected_exc = httpx.HTTPStatusError if status_code >= 300 else ValueError
    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        with pytest.raises(expected_exc) as exc_info:
            await client.upload_tracked_frame(credential, wire, jpeg)

    # Single upload attempt; OutboxDispatcher owns durable retries and RAW gating.
    assert attempts == 1
    _assert_no_secrets_in_error(exc_info.value)


@pytest.mark.anyio
async def test_upload_tracked_frame_transport_error_hides_secrets():
    _, _, credential, _, wire, jpeg, _ = live_evidence_packet()

    def backend(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        with pytest.raises(httpx.TransportError) as exc_info:
            await client.upload_tracked_frame(credential, wire, jpeg)
    _assert_no_secrets_in_error(exc_info.value)


@pytest.mark.anyio
async def test_upload_tracked_frame_preserves_uppercase_wire_and_matches_lowercase_receipt():
    event, context, credential, _, _, jpeg, _ = live_evidence_packet()
    upper_event = copy.deepcopy(event)
    upper_event["eventId"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    upper_event["streamSessionId"] = event["streamSessionId"].upper()
    digest = hashlib.sha256(jpeg).hexdigest()
    upper_event["evidence"] = [
        {
            "kind": "FRAME",
            "uri": (
                f"local://evidence/{event['streamSessionId'].lower()}/"
                f"{event['framePartition']['sequenceNumber']}/"
                f"{upper_event['eventId'].lower()}.jpg"
            ),
            "sha256": digest,
        }
    ]
    envelope = {
        "schemaVersion": "1.0.0",
        "event": upper_event,
        "context": context.model_dump(mode="json", by_alias=True),
    }
    wire = json.dumps(envelope, indent=2).encode("utf-8")
    payload_hash = compute_canonical_payload_hash(upper_event)
    receipt = {
        "schemaVersion": "1.0.0",
        "kind": "EVIDENCE_STORED",
        "eventId": upper_event["eventId"].lower(),
        "payloadHash": payload_hash,
        "bindingHash": context.binding_hash(payload_hash),
        "sha256": digest,
    }

    def backend(request: httpx.Request) -> httpx.Response:
        parts = parse_multipart(request)
        assert json.loads(parts["event"][2].decode("utf-8")) == upper_event
        return httpx.Response(200, json=receipt)

    async with BackendClient(
        "https://backend.example",
        SERVICE_TOKEN,
        transport=httpx.MockTransport(backend),
    ) as client:
        stored = await client.upload_tracked_frame(credential, wire, jpeg)

    assert stored.event_id == upper_event["eventId"].lower()
    assert stored.payload_hash == payload_hash
