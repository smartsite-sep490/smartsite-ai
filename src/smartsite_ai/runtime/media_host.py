"""Explicit HTTP host alongside the existing supervised camera runtime."""

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import SecretStr

from smartsite_ai.camera_media_api import camera_media_router
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.supervisor import (
    RuntimePlan,
    RuntimeReport,
    _finish_cleanup,
    execute_runtime,
)


class _MediaServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The explicit runtime/asyncio runner owns process signals and cleanup.
        yield


async def run_with_camera_preview(
    plan: RuntimePlan,
    observers: dict[str, CameraPreviewObserver],
    service_token: SecretStr,
    *,
    host: str,
    port: int,
    on_bound: Callable[[int], None] | None = None,
) -> RuntimeReport:
    address = ipaddress.ip_address(host)
    if type(port) is not int or not 0 <= port <= 65535:
        raise CameraRuntimeError("Preview bind port is invalid")
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.include_router(camera_media_router(observers, service_token))
    listener = socket.socket(
        socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM
    )
    server: _MediaServer | None = None
    server_task: asyncio.Task[None] | None = None
    runtime_task: asyncio.Task[RuntimeReport] | None = None
    try:
        try:
            listener.bind((str(address), port))
            listener.setblocking(False)
        except OSError:
            raise CameraRuntimeError("Preview bind is unavailable") from None
        server = _MediaServer(
            uvicorn.Config(
                app,
                host=str(address),
                port=port,
                log_level="error",
                access_log=False,
                lifespan="off",
                timeout_graceful_shutdown=1,
            )
        )
        server_task = asyncio.create_task(
            server.serve(sockets=[listener]), name="camera-preview-http"
        )
        async with asyncio.timeout(2):
            while not server.started:
                if server_task.done():
                    await server_task
                    raise CameraRuntimeError("Preview host stopped before startup")
                await asyncio.sleep(0.01)
        if on_bound is not None:
            on_bound(listener.getsockname()[1])
        runtime_task = asyncio.create_task(execute_runtime(plan), name="camera-runtime")
        done, _ = await asyncio.wait(
            {server_task, runtime_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done:
            server_task.result()
            if not runtime_task.done():
                raise CameraRuntimeError("Preview host stopped during camera operation")
        return runtime_task.result()
    finally:
        if server is not None:
            server.should_exit = True
        try:
            if runtime_task is not None:
                if not runtime_task.done():
                    runtime_task.cancel()
                await _finish_cleanup(asyncio.gather(runtime_task, return_exceptions=True))
        finally:
            try:
                if server_task is not None:
                    await _finish_cleanup(server_task)
            finally:
                listener.close()
