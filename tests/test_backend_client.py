import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from smartsite_ai.config import Settings
from smartsite_ai.domain.observations import (
    TechnicalObservationEvent,
)
from smartsite_ai.integrations.backend_client import (
    DEFAULT_TIMEOUT,
    BackendClient,
)


def make_sample_event() -> TechnicalObservationEvent:
    return TechnicalObservationEvent.model_validate(
        {
            "eventId": str(uuid4()),
            "schemaVersion": "1.0.0",
            "cameraExternalId": "CAM-01",
            "streamSessionId": str(uuid4()),
            "capturedAt": "2026-09-19T12:00:00Z",
            "frameDimensions": {"width": 1920, "height": 1080},
            "observations": [
                {
                    "type": "PERSON",
                    "trackId": 1,
                    "confidence": 0.95,
                    "boundingBox": {
                        "x1": 0.1,
                        "y1": 0.1,
                        "x2": 0.5,
                        "y2": 0.8,
                        "coordinateSpace": "NORMALIZED_0_1",
                    },
                }
            ],
            "evidence": [],
        }
    )


class SleepTracker:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


# ============================================================================
# 1. Constructor validation and configuration
# ============================================================================


@pytest.mark.parametrize(
    ("url", "token"),
    [
        (None, "secret-token"),
        ("", "secret-token"),
        ("   ", "secret-token"),
        ("http://backend:3000", None),
        ("http://backend:3000", ""),
        ("http://backend:3000", "   "),
        ("http://backend:3000", SecretStr("")),
        ("http://backend:3000", SecretStr("   ")),
    ],
)
def test_constructor_requires_valid_url_and_token(url: Any, token: Any):
    with pytest.raises(ValueError, match="BackendClient requires a non-empty"):
        BackendClient(base_url=url, service_token=token)


def test_constructor_default_timeout_and_retries():
    client = BackendClient("http://backend:3000", "secret-token")
    assert client.timeout == DEFAULT_TIMEOUT
    assert client.timeout.connect == 5.0
    assert client.timeout.read == 10.0
    assert client.timeout.write == 5.0
    assert client.timeout.pool == 5.0
    assert client.max_retries == 3
    assert client.backoff_factor == 0.5


def test_from_settings_fails_when_unconfigured():
    empty_settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="backend_ingestion_url is not configured"):
        BackendClient.from_settings(empty_settings)

    partial_settings = Settings(
        backend_ingestion_url="http://backend:3000",
        _env_file=None,
    )
    with pytest.raises(ValueError, match="backend_service_token is not configured"):
        BackendClient.from_settings(partial_settings)


def test_from_settings_succeeds_with_complete_config():
    settings = Settings(
        backend_ingestion_url="http://backend:3000",
        backend_service_token=SecretStr("my-valid-token"),
        _env_file=None,
    )
    client = BackendClient.from_settings(settings)
    assert client.base_url == "http://backend:3000"
    assert client.service_token.get_secret_value() == "my-valid-token"


# ============================================================================
# 2. Security: secret token masking
# ============================================================================


def test_client_repr_and_str_never_leak_token():
    raw_token = "super-secret-service-token-999"
    client = BackendClient("http://backend:3000", SecretStr(raw_token))

    assert raw_token not in repr(client)
    assert raw_token not in str(client)
    assert "BackendClient(" in repr(client)


# ============================================================================
# 3. Successful dispatch with exact headers and wire payload
# ============================================================================


@pytest.mark.anyio
async def test_successful_event_dispatch():
    raw_token = "secret-token-abc"
    event = make_sample_event()
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(202, json={"status": "ACCEPTED", "eventId": event.event_id})

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    async with BackendClient(
        base_url="http://backend:3000",
        service_token=raw_token,
        transport=transport,
        sleep_func=tracker.sleep,
    ) as client:
        response = await client.post_event(event)

    assert response.status_code == 202
    assert response.json()["status"] == "ACCEPTED"
    assert len(captured_requests) == 1
    assert tracker.delays == []  # No retries/sleep on immediate success

    req = captured_requests[0]
    assert req.method == "POST"
    assert str(req.url) == "http://backend:3000/api/v1/integrations/ai/events"
    assert req.headers["Authorization"] == f"Bearer {raw_token}"
    assert req.headers["Content-Type"] == "application/json"

    sent_body = json.loads(req.content.decode("utf-8"))
    assert sent_body == event.to_wire_dict()


# ============================================================================
# 4. Retry on TransportError (3 retries = 4 attempts, backoff 0.5, 1.0, 2.0s)
# ============================================================================


@pytest.mark.anyio
async def test_retry_on_transport_error_succeeds_on_fourth_attempt():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            raise httpx.ConnectError("Network is down", request=request)
        return httpx.Response(200, json={"status": "OK"})

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    client = BackendClient(
        base_url="http://backend:3000",
        service_token="tok",
        transport=transport,
        sleep_func=tracker.sleep,
    )

    response = await client.post_event(make_sample_event())
    assert response.status_code == 200
    assert call_count == 4
    assert tracker.delays == [0.5, 1.0, 2.0]
    await client.aclose()


@pytest.mark.anyio
async def test_retry_on_transport_error_exhausts_and_raises():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("Socket timed out", request=request)

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    client = BackendClient(
        base_url="http://backend:3000",
        service_token="tok",
        transport=transport,
        sleep_func=tracker.sleep,
    )

    with pytest.raises(httpx.ReadTimeout):
        await client.post_event(make_sample_event())

    assert call_count == 4
    assert tracker.delays == [0.5, 1.0, 2.0]
    await client.aclose()


# ============================================================================
# 5. Retry on 5xx, 408, 429
# ============================================================================


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
@pytest.mark.anyio
async def test_retry_on_retryable_http_statuses(status_code: int):
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            return httpx.Response(status_code, json={"error": f"Temporary {status_code}"})
        return httpx.Response(200, json={"status": "OK"})

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    client = BackendClient(
        base_url="http://backend:3000",
        service_token="tok",
        transport=transport,
        sleep_func=tracker.sleep,
    )

    response = await client.post_event(make_sample_event())
    assert response.status_code == 200
    assert call_count == 4
    assert tracker.delays == [0.5, 1.0, 2.0]
    await client.aclose()


@pytest.mark.anyio
async def test_retry_on_5xx_exhausts_and_raises_http_status_error():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(503, json={"error": "Service Unavailable"})

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    client = BackendClient(
        base_url="http://backend:3000",
        service_token="tok",
        transport=transport,
        sleep_func=tracker.sleep,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.post_event(make_sample_event())

    assert exc_info.value.response.status_code == 503
    assert call_count == 4
    assert tracker.delays == [0.5, 1.0, 2.0]
    await client.aclose()


# ============================================================================
# 6. Non-retryable 4xx errors fail immediately without retry or sleep
# ============================================================================


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 413, 422])
@pytest.mark.anyio
async def test_non_retryable_4xx_fails_immediately_without_retry(status_code: int):
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, json={"error": f"Client error {status_code}"})

    tracker = SleepTracker()
    transport = httpx.MockTransport(handler)

    client = BackendClient(
        base_url="http://backend:3000",
        service_token="tok",
        transport=transport,
        sleep_func=tracker.sleep,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.post_event(make_sample_event())

    assert exc_info.value.response.status_code == status_code
    assert call_count == 1  # Exactly one attempt, no retries!
    assert tracker.delays == []  # Never slept!
    await client.aclose()
