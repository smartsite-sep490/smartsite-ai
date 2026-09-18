import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import SecretStr

from smartsite_ai.config import Settings
from smartsite_ai.domain.observations import TechnicalObservationEvent

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
INGESTION_ENDPOINT_PATH = "/api/v1/integrations/ai/events"


class BackendClient:
    """Async HTTP client for dispatching observation events to the SmartSite backend.

    Guarantees:
    - Precise timeout (connect=5, read=10, write=5, pool=5).
    - Up to 4 total attempts (max_retries=3) with backoff (0.5s, 1.0s, 2.0s).
    - Retries ONLY TransportError, 408, 429, and 5xx.
    - Fails immediately without retry on non-retryable 4xx errors (400, 401, 403, 404, 409, etc.).
    - Never leaks service token in logs, repr, or exception messages.
    """

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
            raise ValueError("BackendClient requires a non-empty base_url")

        if service_token is None:
            raise ValueError("BackendClient requires a non-empty service_token")

        if isinstance(service_token, str):
            token_str = service_token.strip()
            if not token_str:
                raise ValueError("BackendClient requires a non-empty service_token")
            self._service_token = SecretStr(token_str)
        elif isinstance(service_token, SecretStr):
            if not service_token.get_secret_value().strip():
                raise ValueError("BackendClient requires a non-empty service_token")
            self._service_token = service_token
        else:
            raise ValueError("service_token must be a SecretStr or string")

        raw_url = str(base_url).strip().rstrip("/")
        self._base_url = raw_url
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
        *,
        timeout: httpx.Timeout | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep_func: Callable[[float], Awaitable[None]] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> "BackendClient":
        """Instantiates client from application Settings.

        Fails explicitly if ingestion URL or service token is unconfigured.
        """
        if not settings.backend_ingestion_url:
            raise ValueError("Settings.backend_ingestion_url is not configured")
        if not settings.backend_service_token:
            raise ValueError("Settings.backend_service_token is not configured")

        return cls(
            base_url=settings.backend_ingestion_url,
            service_token=settings.backend_service_token,
            timeout=timeout,
            transport=transport,
            sleep_func=sleep_func,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def service_token(self) -> SecretStr:
        return self._service_token

    @property
    def timeout(self) -> httpx.Timeout:
        return self._timeout

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def backoff_factor(self) -> float:
        return self._backoff_factor

    def _resolve_url(self) -> str:
        if self._base_url.endswith(INGESTION_ENDPOINT_PATH):
            return self._base_url
        return f"{self._base_url}{INGESTION_ENDPOINT_PATH}"

    def _get_headers(self) -> dict[str, str]:
        # Build Authorization header dynamically only at request time to avoid plain state
        return {
            "Authorization": f"Bearer {self._service_token.get_secret_value()}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "BackendClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        # Never include service token in string representation
        return (
            f"BackendClient(base_url={self._base_url!r}, "
            f"timeout={self._timeout!r}, max_retries={self._max_retries})"
        )

    async def post_event(
        self,
        event: TechnicalObservationEvent | dict[str, Any],
    ) -> httpx.Response:
        """Dispatches an observation event to the backend ingestion endpoint.

        Retries on TransportError, 408, 429, and 5xx up to max_retries times.
        Immediately calls raise_for_status on all other 4xx errors without retrying.
        """
        if isinstance(event, TechnicalObservationEvent):
            payload = event.to_wire_dict()
        elif isinstance(event, dict):
            payload = event
        else:
            raise TypeError(
                f"event must be TechnicalObservationEvent or dict, got {type(event).__name__}"
            )

        url = self._resolve_url()
        headers = self._get_headers()
        client = await self._get_client()

        total_attempts = self._max_retries + 1
        last_transport_error: httpx.TransportError | None = None

        for attempt in range(total_attempts):
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt < self._max_retries:
                    delay = self._backoff_factor * (2**attempt)
                    await self._sleep(delay)
                    continue
                # Exhausted retries on network/transport error
                raise last_transport_error from None

            status = response.status_code

            # 1. Successful response (2xx)
            if 200 <= status < 300:
                return response

            # 2. Retryable HTTP errors: 408 (Timeout), 429 (Too Many Requests), 5xx
            if status in (408, 429) or (500 <= status < 600):
                if attempt < self._max_retries:
                    delay = self._backoff_factor * (2**attempt)
                    await self._sleep(delay)
                    continue
                # Exhausted retries on retryable HTTP error
                response.raise_for_status()

            # 3. Non-retryable 4xx (400, 401, 403, 404, 409, 413, 422, etc.) or any other error
            response.raise_for_status()

        if last_transport_error is not None:
            raise last_transport_error
        raise RuntimeError("Unexpected termination of request retry loop")

    send_event = post_event
