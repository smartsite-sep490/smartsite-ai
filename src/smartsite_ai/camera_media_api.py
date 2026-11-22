"""Authenticated observer-only NDJSON media surface consumed through the Backend."""

import asyncio
import json
import math
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from typing import Annotated, Never
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import SecretStr
from starlette.types import Receive, Scope, Send

from smartsite_ai.runtime.media_preview import CameraPreviewObserver

_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Accel-Buffering": "no",
}


class _DeadlineStreamingResponse(StreamingResponse):
    """Bound the ASGI send itself, with at most 50ms for a final empty body."""

    def __init__(self, content: AsyncIterator[bytes], duration: float) -> None:
        super().__init__(content, media_type="application/x-ndjson", headers=_HEADERS)
        self._duration = duration

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with asyncio.timeout(self._duration):
                await super().__call__(scope, receive, send)
        except TimeoutError:
            # A stalled viewer must release its dependency slot, not keep the
            # camera observer or other viewers alive indefinitely.
            with suppress(TimeoutError, OSError):
                async with asyncio.timeout(0.05):
                    await send({"type": "http.response.body", "body": b"", "more_body": False})


def camera_media_router(
    observers: Mapping[str, CameraPreviewObserver],
    service_token: SecretStr,
    *,
    max_stream_seconds: float = 600,
    worker_generation_ids: Mapping[str, str] | None = None,
) -> APIRouter:
    token = service_token.get_secret_value()
    if not 32 <= len(token) <= 4096 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("A bounded server-only preview token is required")
    if not math.isfinite(max_stream_seconds) or not 0 < max_stream_seconds <= 600:
        raise ValueError("Preview stream duration must be bounded")
    generations: dict[str, str] = {}
    if worker_generation_ids is not None:
        if not isinstance(worker_generation_ids, Mapping) or len(worker_generation_ids) > 128:
            raise ValueError("Invalid camera generation bindings")
        for camera, generation in worker_generation_ids.items():
            if (
                not isinstance(camera, str)
                or not camera.strip()
                or len(camera) > 128
                or not camera.isprintable()
                or not isinstance(generation, str)
            ):
                raise ValueError("Invalid camera generation binding")
            generations[camera] = str(UUID(generation))
    expected = ("Bearer " + token).encode("ascii")
    active: dict[str, int] = {}
    router = APIRouter()

    def fail(status: int, code: str) -> Never:
        headers = {**_HEADERS}
        if status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        raise HTTPException(status, detail={"code": code}, headers=headers)

    async def resolve(request: Request) -> CameraPreviewObserver:
        authorization = request.headers.getlist("authorization")
        if len(authorization) != 1 or not secrets.compare_digest(
            authorization[0].encode(), expected
        ):
            fail(401, "UNAUTHENTICATED")
        query = request.query_params.multi_items()
        if len(query) != 1 or query[0][0] != "cameraExternalId" or not query[0][1]:
            fail(400, "INVALID_PREVIEW_QUERY")
        camera = query[0][1]
        observer = observers.get(camera)
        if observer is None or observer.camera_external_id != camera:
            fail(404, "CAMERA_NOT_FOUND")
        return observer

    async def reserve(request: Request) -> AsyncIterator[CameraPreviewObserver]:
        observer = await resolve(request)
        camera = observer.camera_external_id
        if active.get(camera, 0) >= 2 or sum(active.values()) >= 4:
            fail(429, "PREVIEW_VIEWER_LIMIT")
        active[camera] = active.get(camera, 0) + 1
        try:
            yield observer
        finally:
            remaining = active[camera] - 1
            if remaining:
                active[camera] = remaining
            else:
                del active[camera]

    @router.get("/v1/realtime/configuration-image", include_in_schema=False)
    async def configuration_image(
        observer: Annotated[CameraPreviewObserver, Depends(resolve)],
    ) -> JSONResponse:
        payload = observer.configuration_image()
        if payload is None:
            fail(503, "CONFIGURATION_IMAGE_UNAVAILABLE")
        headers = dict(_HEADERS)
        if observer.camera_external_id in generations:
            headers["X-SmartSite-Worker-Generation"] = generations[observer.camera_external_id]
        return JSONResponse(payload, headers=headers)

    @router.get("/v1/realtime/preview/latest", include_in_schema=False)
    async def analysis_preview(
        observer: Annotated[CameraPreviewObserver, Depends(resolve)],
    ) -> JSONResponse:
        frame = await observer.analysis.latest() if observer.analysis is not None else None
        if frame is None:
            fail(503, "ANALYZED_PREVIEW_UNAVAILABLE")
        return JSONResponse({"sourceKind": observer.source_kind, "frame": frame}, headers=_HEADERS)

    @router.get("/v1/realtime/video-stream", include_in_schema=False)
    async def video_stream(
        observer: Annotated[CameraPreviewObserver, Depends(reserve)],
    ) -> StreamingResponse:
        if observer.latest() is None:
            fail(503, "CAMERA_PREVIEW_UNAVAILABLE")

        async def frames() -> AsyncIterator[bytes]:
            deadline = asyncio.get_running_loop().time() + max_stream_seconds
            previous: tuple[str, str] | None = None
            while asyncio.get_running_loop().time() < deadline:
                if observers.get(observer.camera_external_id) is not observer:
                    return
                frame = observer.latest()
                if frame is None:
                    return
                identity = frame["sessionId"], frame["sequenceNumber"]
                if identity != previous:
                    yield (json.dumps(frame, separators=(",", ":")) + "\n").encode("ascii")
                    previous = identity
                await asyncio.sleep(0.1)

        return _DeadlineStreamingResponse(frames(), max_stream_seconds)

    return router
