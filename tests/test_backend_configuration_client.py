from typing import Any
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from smartsite_ai.config import Settings
from smartsite_ai.domain.regions import MAX_CAMERA_REGION_PAYLOAD_BYTES
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationConflictError,
    BackendConfigurationHttpError,
    BackendConfigurationNotFoundError,
    BackendConfigurationResponseError,
    BackendConfigurationTransientError,
    ConfigurationNotModifiedResult,
    ConfigurationSnapshotResult,
)

CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")
EXPECTED_URL = (
    "https://backend.example/api/v1/integrations/ai/cameras/"
    "11111111-1111-4111-8111-111111111111/configuration"
)


def configuration_payload(camera_external_id: str = "CAM-GATE-01") -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "configurationVersion": 7,
        "cameraExternalId": camera_external_id,
        "regions": [
            {
                "regionId": "22222222-2222-4222-8222-222222222222",
                "geometryVersion": 3,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0.1, 0.1], [0.8, 0.1], [0.4, 0.8]]},
            }
        ],
    }


class SleepTracker:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)


class CloseTrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"ETag": '"config-v7"'},
            json=configuration_payload(),
        )

    async def aclose(self) -> None:
        self.closed = True


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://backend.example",
        "https://user:password@backend.example",
        "https://backend.example/api/v1",
        "https://backend.example?token=secret",
        "https://backend.example#fragment",
    ],
)
def test_constructor_rejects_unsafe_or_ambiguous_urls(base_url: str):
    with pytest.raises(ValueError, match=r"HTTP\(S\) origin URL"):
        BackendConfigurationClient(base_url, "token")


def test_from_settings_accepts_existing_ingestion_endpoint_without_config_changes():
    client = BackendConfigurationClient.from_settings(
        Settings(
            backend_ingestion_url="https://backend.example/api/v1/integrations/ai/events",
            backend_service_token=SecretStr("token"),
            _env_file=None,
        )
    )

    assert "https://backend.example" in repr(client)
    assert "/events" not in repr(client)


@pytest.mark.anyio
async def test_fetch_200_authenticates_uses_exact_url_and_returns_typed_snapshot():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"ETag": '"config-v7"'},
            json=configuration_payload(),
        )

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.fetch(
            CAMERA_ID,
            expected_camera_external_id="CAM-GATE-01",
        )

    assert isinstance(result, ConfigurationSnapshotResult)
    assert result.kind == "snapshot"
    assert result.etag == '"config-v7"'
    assert result.configuration.configuration_version == 7
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == EXPECTED_URL
    assert requests[0].headers["Authorization"] == "Bearer service-token"
    assert requests[0].headers["Accept"] == "application/json"
    assert "If-None-Match" not in requests[0].headers


@pytest.mark.anyio
async def test_fetch_uses_if_none_match_and_returns_typed_304_result():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(304)

    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.fetch(
            str(CAMERA_ID),
            expected_camera_external_id="CAM-GATE-01",
            etag='W/"config-v7"',
        )

    assert isinstance(result, ConfigurationNotModifiedResult)
    assert result.kind == "not_modified"
    assert result.etag == 'W/"config-v7"'
    assert requests[0].headers["If-None-Match"] == 'W/"config-v7"'


@pytest.mark.anyio
async def test_304_requires_response_etag_to_weakly_match_request_validator():
    responses = iter(
        [
            httpx.Response(304, headers={"ETag": '"config-v7"'}),
            httpx.Response(304, headers={"ETag": '"config-v8"'}),
        ]
    )
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        equivalent = await client.fetch(
            CAMERA_ID,
            expected_camera_external_id="CAM-GATE-01",
            etag='W/"config-v7"',
        )
        assert isinstance(equivalent, ConfigurationNotModifiedResult)
        assert equivalent.etag == '"config-v7"'

        with pytest.raises(BackendConfigurationResponseError, match="does not match"):
            await client.fetch(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
                etag='"config-v7"',
            )


@pytest.mark.anyio
async def test_304_without_conditional_request_is_an_invalid_response():
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(304)),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="conditional request"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.anyio
async def test_200_requires_etag_and_valid_contract_body():
    responses = iter(
        [
            httpx.Response(200, json=configuration_payload()),
            httpx.Response(200, headers={"ETag": '"v7"'}, content=b"not-json"),
        ]
    )
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="ETag"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")
        with pytest.raises(BackendConfigurationResponseError, match="body is invalid"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.anyio
async def test_200_rejects_oversized_declared_content_length_before_body_read():
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={
                    "ETag": '"v7"',
                    "Content-Length": str(MAX_CAMERA_REGION_PAYLOAD_BYTES + 1),
                },
                stream=ChunkStream([b"{}"]),
            )
        ),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="size limit"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.anyio
async def test_200_bounds_chunked_body_without_content_length():
    chunks = [b"x" * MAX_CAMERA_REGION_PAYLOAD_BYTES, b"x"]
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                stream=ChunkStream(chunks),
            )
        ),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="size limit"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.anyio
async def test_200_rejects_one_oversized_chunk_before_copying_it():
    chunks = [b"x" * (MAX_CAMERA_REGION_PAYLOAD_BYTES + 1)]
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                stream=ChunkStream(chunks),
            )
        ),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="size limit"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.anyio
async def test_fetch_rejects_mismatched_camera_external_id():
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json=configuration_payload("OTHER-CAMERA"),
            )
        ),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="does not match"):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(404, BackendConfigurationNotFoundError), (409, BackendConfigurationConflictError)],
)
@pytest.mark.anyio
async def test_semantic_statuses_are_classified_without_retry(
    status_code: int, error_type: type[Exception]
):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    tracker = SleepTracker()
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
    ) as client:
        with pytest.raises(error_type):
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")

    assert calls == 1
    assert tracker.delays == []


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
@pytest.mark.anyio
async def test_transient_http_statuses_use_bounded_retries(status_code: int):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    tracker = SleepTracker()
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=2,
        backoff_factor=0.25,
    ) as client:
        with pytest.raises(BackendConfigurationTransientError) as exc_info:
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")

    assert exc_info.value.status_code == status_code
    assert exc_info.value.attempts == 3
    assert calls == 3
    assert tracker.delays == [0.25, 0.5]


@pytest.mark.anyio
async def test_transport_errors_are_redacted_and_retried():
    raw_token = "service-token-must-never-appear"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(
            f"cannot connect using {raw_token}",
            request=request,
        )

    tracker = SleepTracker()
    client = BackendConfigurationClient(
        "https://backend.example",
        raw_token,
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=1,
    )

    with pytest.raises(BackendConfigurationTransientError) as exc_info:
        await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")

    assert raw_token not in str(exc_info.value)
    assert raw_token not in repr(exc_info.value)
    assert raw_token not in repr(client)
    assert calls == 2
    assert tracker.delays == [0.5]
    await client.aclose()


@pytest.mark.anyio
async def test_transient_failures_can_recover_within_retry_budget():
    statuses = iter([503, 429, 200])
    tracker = SleepTracker()

    def handler(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json=configuration_payload(),
            )
        return httpx.Response(status)

    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=2,
    ) as client:
        result = await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")

    assert isinstance(result, ConfigurationSnapshotResult)
    assert tracker.delays == [0.5, 1.0]


@pytest.mark.anyio
async def test_non_retryable_http_error_is_redacted_and_not_retried():
    tracker = SleepTracker()
    async with BackendConfigurationClient(
        "https://backend.example",
        "private-token",
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="token rejected")),
        sleep_func=tracker.sleep,
    ) as client:
        with pytest.raises(BackendConfigurationHttpError) as exc_info:
            await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")

    assert exc_info.value.status_code == 401
    assert "private-token" not in str(exc_info.value)
    assert "token rejected" not in str(exc_info.value)
    assert tracker.delays == []


@pytest.mark.anyio
async def test_context_manager_closes_owned_http_client_and_transport():
    transport = CloseTrackingTransport()
    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=transport,
    ) as client:
        await client.fetch(CAMERA_ID, expected_camera_external_id="CAM-GATE-01")
        assert transport.closed is False

    assert transport.closed is True


@pytest.mark.anyio
async def test_invalid_inputs_fail_before_network_dispatch():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ValueError, match="valid UUID"):
            await client.fetch("../not-a-camera", expected_camera_external_id="CAM-GATE-01")
        with pytest.raises(ValueError, match="header-safe"):
            await client.fetch(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
                etag='"ok"\r\nAuthorization: leaked',
            )
        with pytest.raises(ValueError, match="header-safe"):
            await client.fetch(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
                etag="unquoted-etag",
            )

    assert calls == 0
