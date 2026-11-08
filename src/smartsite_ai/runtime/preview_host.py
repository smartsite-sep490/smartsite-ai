"""Private loopback observers sharing the explicitly supervised processing owner."""

import asyncio
import socket
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

import uvicorn
from fastapi import FastAPI
from pydantic import SecretStr

from smartsite_ai.configuration_media_api import configuration_media_router
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.media_preview import MediaPreviewHub
from smartsite_ai.processing_worker import ProcessingWorkerResult
from smartsite_ai.runtime.manifest import BoundManifest, CameraRuntimeError
from smartsite_ai.runtime.supervisor import _finish_cleanup


class PreviewSettings(Protocol):
    stream: StreamConfig
    source_kind: str
    resource_id: UUID
    lock_directory: Path
    token: SecretStr
    port: int
    duration_seconds: float
    raw_preview_enabled: bool


class WorkerPort(Protocol):
    async def run(self) -> ProcessingWorkerResult: ...


class ServerPort(Protocol):
    should_exit: bool

    async def serve(self, *, sockets: list[socket.socket]) -> None: ...


class _Server(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        # The enclosing runtime owns signals and drains camera/model resources.
        yield


def _server(app: FastAPI) -> ServerPort:
    return _Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            log_level="error",
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=2,
        )
    )


@dataclass(repr=False)
class _Binding:
    settings: PreviewSettings
    hub: MediaPreviewHub
    lease: ConfigurationProcessLease
    listener: socket.socket | None = None
    stream: StreamWorker | None = None


class PreviewHostGroup:
    """Reserve before model/source construction; release only after source drain."""

    def __init__(
        self,
        bound: BoundManifest,
        settings: Sequence[PreviewSettings],
        *,
        server_factory: Callable[[FastAPI], ServerPort] = _server,
    ) -> None:
        self._server_factory = server_factory
        cameras = {camera.stream_id: camera for camera in bound.cameras}
        self._bindings: dict[str, _Binding] = {}
        self._reserved = False
        resources: set[UUID] = set()
        ports: set[int] = set()
        if len(settings) > 3:
            raise CameraRuntimeError("raw preview configuration is invalid")
        for config in settings:
            camera = cameras.get(config.stream.stream_id)
            token = config.token.get_secret_value()
            if (
                camera is None
                or not config.raw_preview_enabled
                or config.source_kind != "LIVE"
                or not config.stream.is_live
                or not camera.source_config.is_live
                or camera.camera_external_id != config.stream.camera_external_id
                or camera.source_config.source_url != config.stream.source_url
                or config.stream.stream_id in self._bindings
                or config.resource_id in resources
                or (config.port and config.port in ports)
                or not 0 <= config.port <= 65535
                or not 0 < config.duration_seconds <= 3600
                or not 32 <= len(token) <= 4096
                or not token.isascii()
                or any(c.isspace() for c in token)
            ):
                raise CameraRuntimeError("raw preview configuration is invalid")
            resources.add(config.resource_id)
            ports.add(config.port)
            self._bindings[config.stream.stream_id] = _Binding(
                config,
                MediaPreviewHub(config.stream.camera_external_id),
                ConfigurationProcessLease(config.lock_directory, config.resource_id),
            )

    def reserve(self) -> None:
        if self._reserved:
            raise CameraRuntimeError("raw preview reservation is single-use")
        self._reserved = True
        try:
            for binding in self._bindings.values():
                binding.lease.acquire()
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                binding.listener = listener
                listener.bind(("127.0.0.1", binding.settings.port))
                listener.setblocking(False)
        except (OSError, RuntimeError):
            for binding in self._bindings.values():
                if binding.listener is not None:
                    binding.listener.close()
                binding.lease.release()
            raise CameraRuntimeError("raw preview reservation is unavailable") from None

    def binding_status(self) -> list[dict[str, object]]:
        return [
            {
                "cameraExternalId": item.settings.stream.camera_external_id,
                "port": item.listener.getsockname()[1],
                "state": "RAW_PREVIEW_BOUND_NOT_READY",
            }
            for item in self._bindings.values()
            if item.listener is not None
        ]

    def observer_for(self, stream_id: str) -> MediaPreviewHub | None:
        binding = self._bindings.get(stream_id)
        return binding.hub if binding is not None else None

    def wrap(
        self,
        stream: StreamWorker,
        worker: WorkerPort,
        *,
        server_factory: Callable[[FastAPI], ServerPort] | None = None,
    ) -> WorkerPort:
        binding = self._bindings.get(stream.config.stream_id)
        if binding is None:
            return worker
        if not self._reserved or binding.listener is None or binding.stream is not None:
            raise CameraRuntimeError("raw preview owner is unavailable")
        listener = binding.listener
        binding.stream = stream
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        app.include_router(configuration_media_router(stream, binding.hub, binding.settings.token))

        async def run() -> ProcessingWorkerResult:
            server = (server_factory or self._server_factory)(app)

            async def serve() -> None:
                try:
                    await server.serve(sockets=[listener])
                except (Exception, SystemExit):
                    raise CameraRuntimeError("raw preview server failed") from None

            server_task = asyncio.create_task(serve(), name="durable-preview-server")
            worker_task = asyncio.create_task(worker.run(), name="preview-processing-owner")

            async def shutdown() -> None:
                server.should_exit = True
                if not worker_task.done():
                    worker_task.cancel()
                await asyncio.gather(worker_task, server_task, return_exceptions=True)
                await self._finish(binding)

            try:
                done, _ = await asyncio.wait(
                    (worker_task, server_task),
                    timeout=binding.settings.duration_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if server_task in done:
                    server_task.result()
                    raise CameraRuntimeError("raw preview server stopped unexpectedly")
                if worker_task in done:
                    return worker_task.result()
                raise CameraRuntimeError("raw preview host deadline elapsed")
            finally:
                await _finish_cleanup(shutdown())

        class HostedWorker:
            async def run(self) -> ProcessingWorkerResult:
                return await run()

        return HostedWorker()

    async def _finish(self, binding: _Binding) -> None:
        try:
            if binding.stream is not None:
                await binding.stream.stop()
        finally:
            await binding.hub.close()
            if binding.listener is not None:
                binding.listener.close()
        if binding.stream is None or binding.stream.source_released:
            binding.lease.release()
        else:
            # The lease has no destructor: retain OS ownership on unresolved release.
            raise CameraRuntimeError("raw preview source release is unresolved")

    async def close(self) -> None:
        failures = False
        for binding in self._bindings.values():
            try:
                await self._finish(binding)
            except Exception:
                failures = True
        if failures:
            raise CameraRuntimeError("raw preview source release is unresolved")
