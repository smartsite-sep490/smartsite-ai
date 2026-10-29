"""Explicit capture-only runtime assembly; application startup never opens sources."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from pydantic import SecretStr

from smartsite_ai.configuration_image import SourceKind
from smartsite_ai.configuration_image_api import configuration_image_router
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.configuration_capture import (
    ConfigurationCapture,
    ConfigurationSourceLease,
)
from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import FrameSource, SourceConnectionError

SourceFactory = Callable[[StreamConfig], FrameSource]


def _opencv_source(config: StreamConfig) -> FrameSource:
    # Construction is delayed until explicit start has acquired the process lock.
    # This adapter itself lazily imports cv2 only when connect() is called.
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    return OpenCvFrameSource(config)


class _DeferredSource:
    def __init__(self, config: StreamConfig, factory: SourceFactory) -> None:
        self._config, self._factory = config, factory
        self._source: FrameSource | None = None

    @property
    def source_id(self) -> str:
        return self._config.stream_id

    @property
    def is_connected(self) -> bool:
        return self._source is not None and self._source.is_connected

    async def connect(self) -> None:
        source = self._factory(self._config)
        if source.source_id != self.source_id or source.is_connected:
            raise SourceConnectionError("Configuration source identity or state does not match")
        self._source = source
        await self._source.connect()

    async def read_frame(self) -> FrameEnvelope | None:
        if self._source is None:
            raise SourceConnectionError("Configuration source has not been explicitly opened")
        return await self._source.read_frame()

    async def close(self) -> None:
        if self._source is not None:
            await self._source.close()


class ConfigurationCaptureRuntime:
    """One resource, one explicit worker and an observer-only authenticated app.

    No lifespan hook starts capture. An operator must explicitly call start().
    Application/router construction and requests have no source factory access.
    All cooperating runtimes must share the same physical resource UUID/directory;
    mapping aliases and excluding older non-cooperating workers remains a required
    preflight. Never start this while those owners or their native threads exist.
    """

    def __init__(
        self,
        *,
        stream: StreamConfig,
        source_kind: SourceKind,
        resource_id: UUID,
        lock_directory: Path,
        service_token: SecretStr,
        source_factory: SourceFactory = _opencv_source,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        stop_timeout: float = 2.0,
    ) -> None:
        if source_kind not in ("LIVE", "REPLAY") or stream.is_live != (source_kind == "LIVE"):
            raise ValueError("Configuration source kind must match the stream")
        if source_kind == "REPLAY" and not stream.pace_replay:
            raise ValueError("Configuration replay must be explicitly paced")
        self._process_lease = ConfigurationProcessLease(lock_directory, resource_id)
        source = _DeferredSource(stream, source_factory)
        self._capture = ConfigurationCapture(
            ConfigurationSourceLease(source, stream.camera_external_id, source_kind=source_kind),
            clock=clock,
            stop_timeout=stop_timeout,
        )
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.app.include_router(configuration_image_router(self._capture, service_token))
        self._started = False
        self._stop_requested = False

    @property
    def state(self) -> str:
        return self._capture.state

    @property
    def reason(self) -> str | None:
        return self._capture.reason

    async def start(self) -> None:
        if self._stop_requested or (self._started and self.state != "RUNNING"):
            raise RuntimeError("Configuration runtime lifecycle has ended")
        if self._started:
            return
        self._process_lease.acquire()
        try:
            await self._capture.start()
        except BaseException:
            self._process_lease.release()
            raise
        self._started = True

    async def stop(self) -> bool:
        self._stop_requested = True
        released = await self._capture.stop()
        if released:
            self._process_lease.release()
        return released

    async def wait_for_release(self) -> bool:
        """Keep OS ownership while the existing source task drains after stop."""
        if not self._stop_requested:
            raise RuntimeError("Stop must be requested before waiting for release")
        released = await self._capture.wait_for_release()
        if released:
            self._process_lease.release()
        return released
