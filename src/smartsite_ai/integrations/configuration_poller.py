"""Version-aware polling for whole Backend-owned camera configuration bundles."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from time import monotonic
from uuid import UUID

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationConflict,
    RegionConfigurationStore,
)
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationError,
    BackendConfigurationTransientError,
    ConfigurationBundleResult,
    ConfigurationNotModifiedResult,
)


class ConfigurationStaleError(RuntimeError):
    """The last confirmed configuration exceeded its permitted stale window."""


class ConfigurationRegressionError(RuntimeError):
    """A sequential Backend response attempted to reduce configurationVersion."""


class CameraConfigurationPoller:
    """Poll one camera snapshot and atomically apply validated newer versions."""

    def __init__(
        self,
        *,
        client: BackendConfigurationClient,
        store: RegionConfigurationStore,
        camera_id: UUID,
        camera_external_id: str,
        poll_interval_seconds: float = 5.0,
        stale_after_seconds: float = 60.0,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("configuration poll interval must be positive")
        if not math.isfinite(stale_after_seconds) or stale_after_seconds < poll_interval_seconds:
            raise ValueError("configuration stale window must be at least the poll interval")
        if not camera_external_id:
            raise ValueError("camera external ID must not be empty")
        self._client = client
        self._store = store
        self._camera_id = camera_id
        self._camera_external_id = camera_external_id
        self._poll_interval_seconds = poll_interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._clock = monotonic_clock
        self._etag: str | None = None
        self._last_success: float | None = None

    @property
    def etag(self) -> str | None:
        return self._etag

    async def refresh(self) -> RegionConfigurationApplyOutcome | None:
        try:
            result = await self._client.fetch_bundle(
                self._camera_id,
                expected_camera_external_id=self._camera_external_id,
                etag=self._etag,
            )
        except BackendConfigurationTransientError:
            self.ensure_fresh()
            return None
        except BackendConfigurationError:
            self._etag = None
            self._last_success = None
            raise

        now = self._read_clock()
        if isinstance(result, ConfigurationNotModifiedResult):
            if self._last_success is None:
                raise RuntimeError("configuration poller received 304 before an initial snapshot")
            self._etag = result.etag
            self._last_success = now
            return RegionConfigurationApplyOutcome.IDEMPOTENT

        if not isinstance(result, ConfigurationBundleResult):
            self._etag = None
            self._last_success = None
            raise RuntimeError("configuration client returned an unsupported result")
        try:
            outcome = await self._store.apply_bundle(result.bundle)
        except RegionConfigurationConflict:
            self._etag = None
            self._last_success = None
            raise
        if outcome is RegionConfigurationApplyOutcome.STALE:
            self._etag = None
            self._last_success = None
            raise ConfigurationRegressionError(
                "Backend returned an older camera configuration in a sequential poll"
            )
        self._etag = result.etag
        self._last_success = now
        return outcome

    def ensure_fresh(self) -> None:
        if self._last_success is None:
            raise ConfigurationStaleError("camera configuration has never been confirmed")
        age = self._read_clock() - self._last_success
        if age > self._stale_after_seconds:
            raise ConfigurationStaleError("camera configuration exceeded its stale window")

    def _read_clock(self) -> float:
        now = self._clock()
        if not math.isfinite(now) or (self._last_success is not None and now < self._last_success):
            self._etag = None
            self._last_success = None
            raise ConfigurationStaleError(
                "configuration monotonic clock is invalid or moved backwards"
            )
        return now

    async def run(self, stop_event: asyncio.Event) -> None:
        if self._last_success is None:
            await self.refresh()
            self.ensure_fresh()
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                await self.refresh()


__all__ = [
    "CameraConfigurationPoller",
    "ConfigurationRegressionError",
    "ConfigurationStaleError",
]
