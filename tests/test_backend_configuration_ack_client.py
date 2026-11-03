from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from smartsite_ai.domain.configuration_ack import (
    CameraConfigurationAcknowledgement,
    CameraConfigurationAcknowledgementReceipt,
)
from smartsite_ai.domain.regions import MAX_SAFE_INTEGER
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationConflictError,
    BackendConfigurationHttpError,
    BackendConfigurationNotFoundError,
    BackendConfigurationResponseError,
    BackendConfigurationTransientError,
)

MAX_ACK_BYTES = 4096
CAMERA_ID = UUID("11111111-aaaa-4111-8111-11111111111a")
COMMAND_ID = "22222222-bbbb-4222-8222-22222222222b"
OTHER_COMMAND_ID = "99999999-ffff-4999-8999-99999999999f"
GENERATION_ID = "33333333-cccc-4333-8333-33333333333c"
STREAM_SESSION_ID = "44444444-dddd-4444-8444-44444444444d"
BUNDLE_SHA256 = "0123456789abcdef" * 4
GENERATION_SECRET_VALUE = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-_AbCdE"
EXPECTED_ACK_URL = (
    "https://backend.example/api/v1/integrations/ai/cameras/"
    "11111111-aaaa-4111-8111-11111111111a/configuration-ack"
)


def ack_payload(
    *,
    command_id: str = COMMAND_ID,
    worker_generation_id: str = GENERATION_ID,
    stream_session_id: str = STREAM_SESSION_ID,
    configuration_version: int = 7,
    bundle_sha256: str = BUNDLE_SHA256,
    view_revision: int = 3,
    outcome: str = "LOADED",
    reason_code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "commandId": command_id,
        "workerGenerationId": worker_generation_id,
        "streamSessionId": stream_session_id,
        "configurationVersion": configuration_version,
        "bundleSha256": bundle_sha256,
        "viewRevision": view_revision,
        "outcome": outcome,
    }
    if reason_code is not None:
        payload["reasonCode"] = reason_code
    return payload


def receipt_payload(
    *,
    command_id: str = COMMAND_ID,
    configuration_version: int = 7,
    outcome: str = "LOADED",
    received_at: str = "2026-10-27T09:15:00.123Z",
    replayed: bool = False,
) -> dict[str, Any]:
    return {
        "commandId": command_id,
        "configurationVersion": configuration_version,
        "outcome": outcome,
        "receivedAt": received_at,
        "replayed": replayed,
    }


class SleepTracker:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class CloseTrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=receipt_payload())

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_acknowledge_loaded_rejected_and_replayed_use_exact_headers_and_wire_shape():
    requests: list[httpx.Request] = []
    responses = iter(
        [
            httpx.Response(200, json=receipt_payload(outcome="LOADED", replayed=False)),
            httpx.Response(
                200,
                json=receipt_payload(
                    command_id=OTHER_COMMAND_ID, outcome="REJECTED", replayed=False
                ),
            ),
            httpx.Response(
                200,
                json=receipt_payload(
                    command_id=OTHER_COMMAND_ID, outcome="REJECTED", replayed=True
                ),
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(responses)

    loaded_ack = CameraConfigurationAcknowledgement.model_validate(ack_payload(outcome="LOADED"))
    rejected_ack = CameraConfigurationAcknowledgement.model_validate(
        ack_payload(
            command_id=OTHER_COMMAND_ID, outcome="REJECTED", reason_code="REGION_GEOMETRY_INVALID"
        )
    )
    rejected_without_reason_ack = CameraConfigurationAcknowledgement.model_validate(
        ack_payload(outcome="REJECTED")
    )

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-bearer-token",
        transport=httpx.MockTransport(handler),
    ) as client:
        loaded_receipt = await client.acknowledge_configuration(
            str(CAMERA_ID).upper(),
            generation_secret=SecretStr(GENERATION_SECRET_VALUE),
            acknowledgement=loaded_ack,
        )
        rejected_receipt = await client.acknowledge_configuration(
            CAMERA_ID,
            generation_secret=SecretStr(GENERATION_SECRET_VALUE),
            acknowledgement=rejected_ack,
        )
        replayed_receipt = await client.acknowledge_configuration(
            CAMERA_ID,
            generation_secret=SecretStr(GENERATION_SECRET_VALUE),
            acknowledgement=rejected_ack,
        )

    assert isinstance(loaded_receipt, CameraConfigurationAcknowledgementReceipt)
    assert loaded_receipt.command_id == COMMAND_ID
    assert loaded_receipt.configuration_version == 7
    assert loaded_receipt.outcome == "LOADED"
    assert loaded_receipt.received_at == "2026-10-27T09:15:00.123Z"
    assert loaded_receipt.replayed is False

    assert rejected_receipt.outcome == "REJECTED"
    assert rejected_receipt.replayed is False
    assert replayed_receipt.outcome == "REJECTED"
    assert replayed_receipt.replayed is True

    with pytest.raises((ValidationError, FrozenInstanceError, TypeError)):
        loaded_ack.outcome = "REJECTED"  # type: ignore[misc]
    with pytest.raises((ValidationError, FrozenInstanceError, TypeError)):
        loaded_receipt.replayed = True  # type: ignore[misc]

    assert len(requests) == 3
    for request in requests:
        assert request.method == "POST"
        assert str(request.url) == EXPECTED_ACK_URL
        assert request.headers["Authorization"] == "Bearer service-bearer-token"
        assert request.headers["Accept"] == "application/json"
        assert request.headers.get_list("x-worker-generation-secret") == [GENERATION_SECRET_VALUE]

    loaded_wire = json.loads(requests[0].content.decode("utf-8"))
    assert loaded_wire == ack_payload(outcome="LOADED")
    assert "reasonCode" not in loaded_wire

    rejected_wire = json.loads(requests[1].content.decode("utf-8"))
    assert rejected_wire == ack_payload(
        command_id=OTHER_COMMAND_ID, outcome="REJECTED", reason_code="REGION_GEOMETRY_INVALID"
    )

    assert requests[2].content == requests[1].content
    assert "reasonCode" not in rejected_without_reason_ack.model_dump(
        by_alias=True, exclude_none=True
    )


def test_acknowledgement_domain_model_enforces_strict_contract_invariants():
    extra_key = ack_payload()
    extra_key["unexpected"] = "nope"
    with pytest.raises((ValidationError, ValueError)):
        CameraConfigurationAcknowledgement.model_validate(extra_key)

    loaded_with_reason = ack_payload(outcome="LOADED", reason_code="NOT_ALLOWED_ON_LOADED")
    with pytest.raises((ValidationError, ValueError)):
        CameraConfigurationAcknowledgement.model_validate(loaded_with_reason)

    for bad_reason in ("lowercase_reason", "1LEADING_DIGIT", "", "A" * 65, "BAD-HYPHEN"):
        with pytest.raises((ValidationError, ValueError)):
            CameraConfigurationAcknowledgement.model_validate(
                ack_payload(outcome="REJECTED", reason_code=bad_reason)
            )

    for uuid_field in ("commandId", "workerGenerationId", "streamSessionId"):
        for bad_uuid in (COMMAND_ID.upper(), "not-a-uuid"):
            candidate = ack_payload()
            candidate[uuid_field] = bad_uuid
            with pytest.raises((ValidationError, ValueError)):
                CameraConfigurationAcknowledgement.model_validate(candidate)

    for int_field in ("configurationVersion", "viewRevision"):
        for bad_int in (0, -1, True, MAX_SAFE_INTEGER + 1):
            candidate = ack_payload()
            candidate[int_field] = bad_int
            with pytest.raises((ValidationError, ValueError)):
                CameraConfigurationAcknowledgement.model_validate(candidate)

    for bad_hash in (BUNDLE_SHA256.upper(), "a" * 63, "g" * 64):
        with pytest.raises((ValidationError, ValueError)):
            CameraConfigurationAcknowledgement.model_validate(ack_payload(bundle_sha256=bad_hash))


@pytest.mark.anyio
async def test_invalid_camera_scope_or_generation_secret_fails_before_network_dispatch():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=receipt_payload())

    valid_ack = CameraConfigurationAcknowledgement.model_validate(ack_payload())
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
    ) as client:
        for bad_camera_id in ("../not-a-uuid", "not-a-camera-uuid", ""):
            with pytest.raises((ValueError, TypeError)):
                await client.acknowledge_configuration(
                    bad_camera_id,
                    generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                    acknowledgement=valid_ack,
                )

        for bad_secret in (
            SecretStr(""),
            SecretStr("short-secret"),
            SecretStr("A" * 42),
            SecretStr("A" * 44),
            SecretStr("A" * 42 + "+"),
            SecretStr("A" * 42 + "/"),
            SecretStr("A" * 42 + "="),
            SecretStr("A" * 41 + "\r\n"),
        ):
            with pytest.raises((ValueError, TypeError)):
                await client.acknowledge_configuration(
                    CAMERA_ID,
                    generation_secret=bad_secret,
                    acknowledgement=valid_ack,
                )

        with pytest.raises((ValueError, TypeError)):
            await client.acknowledge_configuration(
                CAMERA_ID,
                generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                acknowledgement=ack_payload(),  # type: ignore[arg-type]
            )

    assert calls == 0


@pytest.mark.anyio
async def test_transient_failures_retry_with_identical_command_body_and_exhaust_cleanly():
    bodies: list[bytes] = []
    tracker = SleepTracker()
    valid_ack = CameraConfigurationAcknowledgement.model_validate(
        ack_payload(outcome="REJECTED", reason_code="MODEL_INPUT_INVALID")
    )

    def recover_handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        attempt = len(bodies)
        if attempt == 1:
            raise httpx.ConnectError("transient socket reset", request=request)
        if attempt == 2:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(
            200,
            json=receipt_payload(outcome="REJECTED", replayed=True),
        )

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token",
        transport=httpx.MockTransport(recover_handler),
        sleep_func=tracker.sleep,
        max_retries=2,
        backoff_factor=0.25,
    ) as client:
        receipt = await client.acknowledge_configuration(
            CAMERA_ID,
            generation_secret=SecretStr(GENERATION_SECRET_VALUE),
            acknowledgement=valid_ack,
        )

    assert receipt.replayed is True
    assert len(bodies) == 3
    assert bodies[0] == bodies[1] == bodies[2]
    assert tracker.delays == [0.25, 0.5]

    exhaust_calls = 0
    exhaust_tracker = SleepTracker()

    def always_429(_: httpx.Request) -> httpx.Response:
        nonlocal exhaust_calls
        exhaust_calls += 1
        return httpx.Response(429, text="rate limited")

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token",
        transport=httpx.MockTransport(always_429),
        sleep_func=exhaust_tracker.sleep,
        max_retries=2,
        backoff_factor=0.25,
    ) as client:
        with pytest.raises(BackendConfigurationTransientError) as exc_info:
            await client.acknowledge_configuration(
                CAMERA_ID,
                generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                acknowledgement=valid_ack,
            )

    assert exc_info.value.status_code == 429
    assert exc_info.value.attempts == 3
    assert exhaust_calls == 3
    assert exhaust_tracker.delays == [0.25, 0.5]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, BackendConfigurationHttpError),
        (403, BackendConfigurationHttpError),
        (404, BackendConfigurationNotFoundError),
        (409, BackendConfigurationConflictError),
        (422, BackendConfigurationHttpError),
    ],
)
@pytest.mark.anyio
async def test_auth_and_semantic_errors_are_not_retried_and_redact_secrets(
    status_code: int,
    error_type: type[Exception],
):
    calls = 0
    tracker = SleepTracker()
    service_token = "service-bearer-secret-do-not-leak"
    raw_error_body = "backend-internal-diagnostic-must-not-leak"
    valid_ack = CameraConfigurationAcknowledgement.model_validate(ack_payload())

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text=raw_error_body)

    async with BackendConfigurationClient(
        "https://backend.example",
        service_token,
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=3,
    ) as client:
        with pytest.raises(error_type) as exc_info:
            await client.acknowledge_configuration(
                CAMERA_ID,
                generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                acknowledgement=valid_ack,
            )

    assert calls == 1
    assert tracker.delays == []
    if isinstance(exc_info.value, BackendConfigurationHttpError):
        assert exc_info.value.status_code == status_code
    assert service_token not in str(exc_info.value)
    assert service_token not in repr(exc_info.value)
    assert GENERATION_SECRET_VALUE not in str(exc_info.value)
    assert GENERATION_SECRET_VALUE not in repr(exc_info.value)
    assert raw_error_body not in str(exc_info.value)


@pytest.mark.anyio
async def test_bounded_4096_response_and_matching_receipt_validation_with_redaction():
    service_token = "private-service-token-value"
    leaked_body_secret = "SENSITIVE_RECEIPT_BODY_SECRET"
    valid_ack = CameraConfigurationAcknowledgement.model_validate(ack_payload())

    extra_field_receipt = receipt_payload()
    extra_field_receipt["unexpected"] = leaked_body_secret

    responses = iter(
        [
            httpx.Response(
                200,
                headers={"Content-Length": str(MAX_ACK_BYTES + 1)},
                stream=ChunkStream([b"{}"]),
            ),
            httpx.Response(
                200,
                stream=ChunkStream([b"x" * MAX_ACK_BYTES, b"x"]),
            ),
            httpx.Response(
                200,
                stream=ChunkStream([b"x" * (MAX_ACK_BYTES + 1)]),
            ),
            httpx.Response(200, content=b"not-valid-json"),
            httpx.Response(200, json=extra_field_receipt),
            httpx.Response(200, json=receipt_payload(command_id=OTHER_COMMAND_ID)),
            httpx.Response(200, json=receipt_payload(configuration_version=8)),
            httpx.Response(200, json=receipt_payload(outcome="REJECTED")),
            httpx.Response(200, json=receipt_payload(received_at="2026-10-27T09:15:00Z")),
        ]
    )

    async with BackendConfigurationClient(
        "https://backend.example",
        service_token,
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        for _ in range(9):
            with pytest.raises(BackendConfigurationResponseError) as exc_info:
                await client.acknowledge_configuration(
                    CAMERA_ID,
                    generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                    acknowledgement=valid_ack,
                )
            assert service_token not in str(exc_info.value)
            assert GENERATION_SECRET_VALUE not in str(exc_info.value)
            assert leaked_body_secret not in str(exc_info.value)
            assert leaked_body_secret not in repr(exc_info.value)
            assert exc_info.value.__cause__ is None


@pytest.mark.anyio
async def test_cancellation_propagates_during_backoff_and_context_manager_closes_transport():
    calls = 0
    valid_ack = CameraConfigurationAcknowledgement.model_validate(ack_payload())

    async def cancelling_sleep(_: float) -> None:
        raise asyncio.CancelledError

    def always_503(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token",
        transport=httpx.MockTransport(always_503),
        sleep_func=cancelling_sleep,
        max_retries=3,
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.acknowledge_configuration(
                CAMERA_ID,
                generation_secret=SecretStr(GENERATION_SECRET_VALUE),
                acknowledgement=valid_ack,
            )

    assert calls == 1

    transport = CloseTrackingTransport()
    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token",
        transport=transport,
    ) as client:
        receipt = await client.acknowledge_configuration(
            CAMERA_ID,
            generation_secret=SecretStr(GENERATION_SECRET_VALUE),
            acknowledgement=valid_ack,
        )
        assert receipt.command_id == COMMAND_ID
        assert transport.closed is False

    assert transport.closed is True
