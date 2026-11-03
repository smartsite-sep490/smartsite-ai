from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any
from uuid import UUID

import httpx
import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import (
    MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES,
    CameraConfigurationBundle,
)
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationConflictError,
    BackendConfigurationHttpError,
    BackendConfigurationNotFoundError,
    BackendConfigurationResponseError,
    ConfigurationBundleResult,
    ConfigurationNotModifiedResult,
)

CAMERA_ID = UUID("11111111-aaaa-4111-8111-11111111111a")
OTHER_CAMERA_ID = UUID("99999999-cccc-4999-8999-99999999999c")
REGION_ID = "22222222-bbbb-4222-8222-22222222222b"
ZONE_ID = "33333333-dddd-4333-8333-33333333333d"
EXPECTED_BUNDLE_URL = (
    "https://backend.example/api/v1/integrations/ai/cameras/"
    "11111111-aaaa-4111-8111-11111111111a/configuration-bundle"
)


def bundle_payload(
    *,
    camera_id: str = str(CAMERA_ID),
    camera_external_id: str = "CAM-GATE-01",
    configuration_version: int = 7,
    view_revision: int = 3,
    required_ppe: list[str] | None = None,
    bundle_sha256: str | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "cameraId": camera_id,
        "cameraExternalId": camera_external_id,
        "configurationVersion": configuration_version,
        "viewRevision": view_revision,
        "issuedAt": "2026-10-27T09:00:00.000Z",
        "regions": [
            {
                "regionId": REGION_ID,
                "geometryVersion": 2,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0.1, 0.1], [0.8, 0.1], [0.4, 0.8]]},
                "zoneId": ZONE_ID,
                "policyRevision": 4,
                "requiredPpe": (
                    ["HARD_HAT", "SAFETY_VEST", "GOGGLES"] if required_ppe is None else required_ppe
                ),
            }
        ],
    }
    digest = compute_canonical_payload_hash(content) if bundle_sha256 is None else bundle_sha256
    return {
        **content,
        "bundleSha256": digest,
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


@pytest.mark.anyio
async def test_fetch_bundle_200_uses_exact_url_and_returns_frozen_whole_bundle():
    requests: list[httpx.Request] = []
    wire = bundle_payload()
    etag = f'W/"cfg-v7-r3-{wire["bundleSha256"]}"'

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"ETag": etag}, json=wire)

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-secret-token",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.fetch_bundle(
            str(CAMERA_ID).upper(),
            expected_camera_external_id="CAM-GATE-01",
        )

    assert isinstance(result, ConfigurationBundleResult)
    assert result.kind == "bundle"
    assert result.etag == etag
    assert isinstance(result.bundle, CameraConfigurationBundle)
    assert result.bundle.camera_id == str(CAMERA_ID)
    assert result.bundle.camera_external_id == "CAM-GATE-01"
    assert result.bundle.configuration_version == 7
    assert result.bundle.view_revision == 3
    assert result.bundle.bundle_sha256 == wire["bundleSha256"]
    assert result.bundle.regions[0].zone_id == ZONE_ID
    assert result.bundle.regions[0].policy_revision == 4
    assert result.bundle.regions[0].required_ppe == (
        "HARD_HAT",
        "SAFETY_VEST",
        "GOGGLES",
    )

    with pytest.raises(FrozenInstanceError):
        result.etag = '"mutated"'  # type: ignore[misc]

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == EXPECTED_BUNDLE_URL
    assert requests[0].headers["Authorization"] == "Bearer service-secret-token"
    assert requests[0].headers["Accept"] == "application/json"
    assert "If-None-Match" not in requests[0].headers


@pytest.mark.anyio
async def test_fetch_bundle_304_validates_etag_and_never_fabricates_bundle():
    requests: list[httpx.Request] = []
    responses = iter(
        [
            httpx.Response(304),
            httpx.Response(304, headers={"ETag": '"cfg-v7-r3"'}),
            httpx.Response(304, headers={"ETag": '"cfg-v7-r4"'}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(responses)

    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(handler),
    ) as client:
        first = await client.fetch_bundle(
            CAMERA_ID,
            expected_camera_external_id="CAM-GATE-01",
            etag='W/"cfg-v7-r3"',
        )
        assert isinstance(first, ConfigurationNotModifiedResult)
        assert not hasattr(first, "bundle")
        assert first.kind == "not_modified"
        assert first.etag == 'W/"cfg-v7-r3"'
        assert requests[0].headers["If-None-Match"] == 'W/"cfg-v7-r3"'

        second = await client.fetch_bundle(
            CAMERA_ID,
            expected_camera_external_id="CAM-GATE-01",
            etag='W/"cfg-v7-r3"',
        )
        assert isinstance(second, ConfigurationNotModifiedResult)
        assert second.etag == '"cfg-v7-r3"'

        with pytest.raises(BackendConfigurationResponseError, match="does not match"):
            await client.fetch_bundle(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
                etag='"cfg-v7-r3"',
            )

    async with BackendConfigurationClient(
        "https://backend.example",
        "token",
        transport=httpx.MockTransport(lambda _: httpx.Response(304)),
    ) as client:
        with pytest.raises(BackendConfigurationResponseError, match="conditional request"):
            await client.fetch_bundle(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
            )


@pytest.mark.anyio
async def test_fetch_bundle_rejects_mismatched_scope_invalid_hash_oversized_and_redacts_payload():
    leaked_payload_secret = "SENSITIVE_INTERNAL_PAYLOAD_VALUE_DO_NOT_LEAK"
    responses = iter(
        [
            httpx.Response(200, json=bundle_payload()),
            httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json=bundle_payload(camera_id=str(OTHER_CAMERA_ID)),
            ),
            httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json=bundle_payload(camera_external_id="OTHER-CAMERA"),
            ),
            httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json=bundle_payload(bundle_sha256="a" * 64),
            ),
            httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                json={"secret": leaked_payload_secret},
            ),
            httpx.Response(
                200,
                headers={
                    "ETag": '"v7"',
                    "Content-Length": str(MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES + 1),
                },
                stream=ChunkStream([b"{}"]),
            ),
            httpx.Response(
                200,
                headers={"ETag": '"v7"'},
                stream=ChunkStream([b"x" * MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES, b"x"]),
            ),
        ]
    )

    async with BackendConfigurationClient(
        "https://backend.example",
        "private-service-token",
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        for expected_match in (
            "ETag",
            "cameraId does not match",
            "cameraExternalId does not match",
            "body is invalid",
            "body is invalid",
            "size limit",
            "size limit",
        ):
            with pytest.raises(BackendConfigurationResponseError, match=expected_match) as exc_info:
                await client.fetch_bundle(
                    CAMERA_ID,
                    expected_camera_external_id="CAM-GATE-01",
                )
            assert "private-service-token" not in str(exc_info.value)
            assert leaked_payload_secret not in str(exc_info.value)
            assert leaked_payload_secret not in repr(exc_info.value)
            assert exc_info.value.__cause__ is None


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
async def test_fetch_bundle_does_not_retry_auth_or_semantic_errors_and_redacts_body(
    status_code: int,
    error_type: type[Exception],
):
    calls = 0
    tracker = SleepTracker()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text="raw-backend-error-secret")

    async with BackendConfigurationClient(
        "https://backend.example",
        "super-secret-bearer",
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=3,
    ) as client:
        with pytest.raises(error_type) as exc_info:
            await client.fetch_bundle(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
            )

    assert calls == 1
    assert tracker.delays == []
    assert "super-secret-bearer" not in str(exc_info.value)
    assert "raw-backend-error-secret" not in str(exc_info.value)


@pytest.mark.anyio
async def test_fetch_bundle_retries_transient_failures_and_preserves_cancellation():
    attempts = 0
    tracker = SleepTracker()
    wire = bundle_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("transient network error", request=request)
        if attempts == 2:
            return httpx.Response(503, text="service busy")
        return httpx.Response(200, headers={"ETag": '"v7"'}, json=wire)

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token-value",
        transport=httpx.MockTransport(handler),
        sleep_func=tracker.sleep,
        max_retries=2,
        backoff_factor=0.25,
    ) as client:
        recovered = await client.fetch_bundle(
            CAMERA_ID,
            expected_camera_external_id="CAM-GATE-01",
        )

    assert isinstance(recovered, ConfigurationBundleResult)
    assert attempts == 3
    assert tracker.delays == [0.25, 0.5]

    cancel_calls = 0

    async def cancelling_sleep(_: float) -> None:
        raise asyncio.CancelledError

    def always_503(_: httpx.Request) -> httpx.Response:
        nonlocal cancel_calls
        cancel_calls += 1
        return httpx.Response(503)

    async with BackendConfigurationClient(
        "https://backend.example",
        "service-token-value",
        transport=httpx.MockTransport(always_503),
        sleep_func=cancelling_sleep,
        max_retries=3,
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.fetch_bundle(
                CAMERA_ID,
                expected_camera_external_id="CAM-GATE-01",
            )

    assert cancel_calls == 1
