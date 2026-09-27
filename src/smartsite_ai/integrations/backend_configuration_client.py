import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import SecretStr

from smartsite_ai.config import Settings
from smartsite_ai.domain.regions import (
    MAX_CAMERA_REGION_PAYLOAD_BYTES,
    CameraRegionConfiguration,
)
from smartsite_ai.integrations.backend_client import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    INGESTION_ENDPOINT_PATH,
)

CONFIGURATION_ENDPOINT_PREFIX = "/api/v1/integrations/ai/cameras"
MAX_ETAG_LENGTH = 512
_ETAG_PATTERN = re.compile(r'(?:W/)?"[\x21\x23-\x7e]*"\Z')


class BackendConfigurationError(RuntimeError):
    """Base error that never retains request credentials or response payloads."""


class BackendConfigurationNotFoundError(BackendConfigurationError):
    """The camera configuration is absent or outside the service allowlist."""


class BackendConfigurationConflictError(BackendConfigurationError):
    """The camera exists but cannot currently provide an active configuration."""


class BackendConfigurationHttpError(BackendConfigurationError):
    """A non-retryable HTTP response not covered by a semantic error."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Backend configuration request failed with HTTP {status_code}")


class BackendConfigurationTransientError(BackendConfigurationError):
    """A retryable transport or HTTP failure remained after bounded attempts."""

    def __init__(self, *, attempts: int, status_code: int | None = None) -> None:
        self.attempts = attempts
        self.status_code = status_code
        reason = "transport failure" if status_code is None else f"HTTP {status_code}"
        super().__init__(
            f"Backend configuration request exhausted {attempts} attempts after {reason}"
        )


class BackendConfigurationResponseError(BackendConfigurationError):
    """The Backend returned a successful status with an invalid contract response."""


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshotResult:
    configuration: CameraRegionConfiguration
    etag: str
    kind: Literal["snapshot"] = "snapshot"


@dataclass(frozen=True, slots=True)
class ConfigurationNotModifiedResult:
    etag: str
    kind: Literal["not_modified"] = "not_modified"


ConfigurationFetchResult = ConfigurationSnapshotResult | ConfigurationNotModifiedResult


def _validated_etag(value: str | None, *, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_ETAG_LENGTH
        or _ETAG_PATTERN.fullmatch(value) is None
    ):
        raise BackendConfigurationResponseError(f"{source} ETag is missing or invalid")
    return value


def _validated_expected_external_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("expected_camera_external_id must be a valid camera external ID")
    if "\x00" in value or any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("expected_camera_external_id must be a valid camera external ID")
    return value


def _weak_etag_equal(left: str, right: str) -> bool:
    def opaque(value: str) -> str:
        return value[2:] if value.startswith("W/") else value

    return opaque(left) == opaque(right)


async def _read_bounded_configuration_body(response: httpx.Response) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise BackendConfigurationResponseError(
                "Backend configuration response Content-Length is invalid"
            ) from exc
        if declared_length < 0 or declared_length > MAX_CAMERA_REGION_PAYLOAD_BYTES:
            raise BackendConfigurationResponseError(
                "Backend configuration response body exceeds the size limit"
            )

    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > MAX_CAMERA_REGION_PAYLOAD_BYTES:
            raise BackendConfigurationResponseError(
                "Backend configuration response body exceeds the size limit"
            )
        body.extend(chunk)
    return bytes(body)


class BackendConfigurationClient:
    """Fetch versioned camera-region snapshots from the SmartSite Backend."""

    def __init__(
        self,
        base_url: str | None,
        service_token: SecretStr | str | None,
        *,
        timeout: httpx.Timeout | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep_func: Callable[[float], Awaitable[None]] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        if not base_url or not str(base_url).strip():
            raise ValueError("BackendConfigurationClient requires a non-empty base_url")
        if service_token is None:
            raise ValueError("BackendConfigurationClient requires a non-empty service_token")

        if isinstance(service_token, str):
            token = service_token.strip()
            if not token:
                raise ValueError("BackendConfigurationClient requires a non-empty service_token")
            self._service_token = SecretStr(token)
        elif isinstance(service_token, SecretStr):
            if not service_token.get_secret_value().strip():
                raise ValueError("BackendConfigurationClient requires a non-empty service_token")
            self._service_token = service_token
        else:
            raise ValueError("service_token must be a SecretStr or string")

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")

        raw_url = str(base_url).strip().rstrip("/")
        parsed = urlsplit(raw_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
            or parsed.path not in ("", INGESTION_ENDPOINT_PATH)
        ):
            raise ValueError(
                "BackendConfigurationClient base_url must be an HTTP(S) origin URL "
                "or exact ingestion endpoint"
            )

        self._origin = f"{parsed.scheme}://{parsed.netloc}"
        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._transport = transport
        self._sleep = sleep_func if sleep_func is not None else asyncio.sleep
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **kwargs: Any,
    ) -> "BackendConfigurationClient":
        if not settings.backend_ingestion_url:
            raise ValueError("Settings.backend_ingestion_url is not configured")
        if not settings.backend_service_token:
            raise ValueError("Settings.backend_service_token is not configured")
        return cls(settings.backend_ingestion_url, settings.backend_service_token, **kwargs)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "BackendConfigurationClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"BackendConfigurationClient(base_url={self._origin!r}, "
            f"timeout={self._timeout!r}, max_retries={self._max_retries})"
        )

    async def fetch(
        self,
        camera_id: UUID | str,
        *,
        expected_camera_external_id: str,
        etag: str | None = None,
    ) -> ConfigurationFetchResult:
        try:
            normalized_camera_id = str(UUID(str(camera_id)))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("camera_id must be a valid UUID") from exc
        expected_external_id = _validated_expected_external_id(expected_camera_external_id)

        headers = {
            "Authorization": f"Bearer {self._service_token.get_secret_value()}",
            "Accept": "application/json",
        }
        if etag is not None:
            try:
                headers["If-None-Match"] = _validated_etag(etag, source="request")
            except BackendConfigurationResponseError as exc:
                raise ValueError("etag must be a non-empty, header-safe value") from exc

        url = f"{self._origin}{CONFIGURATION_ENDPOINT_PREFIX}/{normalized_camera_id}/configuration"
        client = await self._get_client()
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    status = response.status_code
                    if status == 200:
                        response_etag = response.headers.get("ETag")
                        validated_etag = _validated_etag(response_etag, source="response")
                        payload = await _read_bounded_configuration_body(response)
                        try:
                            configuration = CameraRegionConfiguration.from_wire_bytes(payload)
                        except (TypeError, ValueError) as exc:
                            raise BackendConfigurationResponseError(
                                "Backend configuration response body is invalid"
                            ) from exc
                        if configuration.camera_external_id != expected_external_id:
                            raise BackendConfigurationResponseError(
                                "Backend configuration cameraExternalId does not match "
                                "the expected camera"
                            )
                        return ConfigurationSnapshotResult(configuration, validated_etag)

                    if status == 304:
                        if etag is None:
                            raise BackendConfigurationResponseError(
                                "Backend returned 304 without a conditional request"
                            )
                        response_etag = response.headers.get("ETag")
                        if response_etag is None:
                            return ConfigurationNotModifiedResult(headers["If-None-Match"])
                        validated_etag = _validated_etag(response_etag, source="response")
                        if not _weak_etag_equal(validated_etag, headers["If-None-Match"]):
                            raise BackendConfigurationResponseError(
                                "Backend 304 ETag does not match the requested validator"
                            )
                        return ConfigurationNotModifiedResult(validated_etag)

                    if status == 404:
                        raise BackendConfigurationNotFoundError(
                            "Backend camera configuration was not found or is not allowed"
                        )
                    if status == 409:
                        raise BackendConfigurationConflictError(
                            "Backend camera configuration is currently unavailable due to "
                            "camera state"
                        )
                    if status in (408, 429) or 500 <= status < 600:
                        if attempt < self._max_retries:
                            await self._sleep(self._backoff_factor * (2**attempt))
                            continue
                        raise BackendConfigurationTransientError(
                            attempts=attempts, status_code=status
                        )
                    raise BackendConfigurationHttpError(status)
            except httpx.TransportError:
                if attempt < self._max_retries:
                    await self._sleep(self._backoff_factor * (2**attempt))
                    continue
                raise BackendConfigurationTransientError(attempts=attempts) from None

        raise RuntimeError("Unexpected termination of configuration request retry loop")


__all__ = [
    "BackendConfigurationClient",
    "BackendConfigurationConflictError",
    "BackendConfigurationError",
    "BackendConfigurationHttpError",
    "BackendConfigurationNotFoundError",
    "BackendConfigurationResponseError",
    "BackendConfigurationTransientError",
    "ConfigurationFetchResult",
    "ConfigurationNotModifiedResult",
    "ConfigurationSnapshotResult",
]
