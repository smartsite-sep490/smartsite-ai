"""Bounded display observers of an explicitly started, model-free capture owner."""

import asyncio
import json
import secrets
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import SecretStr
from starlette.background import BackgroundTask

from smartsite_ai.ingestion.configuration_capture import ConfigurationCapture
from smartsite_ai.media_preview import MediaPreviewHub

_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def configuration_media_router(
    capture: ConfigurationCapture, hub: MediaPreviewHub, service_token: SecretStr
) -> APIRouter:
    token = service_token.get_secret_value()
    if not 32 <= len(token) <= 4096 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("A bounded server-only preview service token is required")
    expected = f"Bearer {token}".encode("ascii")
    router = APIRouter()
    viewers = 0

    def error(status: int, code: str) -> JSONResponse:
        headers = {**_HEADERS}
        if status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        if status == 429:
            headers["Retry-After"] = "2"
        return JSONResponse({"code": code}, status_code=status, headers=headers)

    @router.get("/v1/realtime/video-stream", include_in_schema=False)
    async def video_stream(request: Request) -> Response:
        nonlocal viewers
        authorization = request.headers.getlist("authorization")
        if len(authorization) != 1 or not secrets.compare_digest(
            authorization[0].encode("utf-8"), expected
        ):
            return error(401, "UNAUTHENTICATED")
        query = request.query_params.multi_items()
        if len(query) != 1 or query[0][0] != "cameraExternalId" or not query[0][1]:
            return error(400, "INVALID_PREVIEW_QUERY")
        if query[0][1] != capture.camera_external_id:
            return error(404, "CAMERA_NOT_FOUND")
        frame = hub.latest() if capture.observer_available else None
        if frame is None:
            return error(503, "MEDIA_PREVIEW_UNAVAILABLE")
        if viewers >= 2:
            return error(429, "MEDIA_PREVIEW_BUSY")
        viewers += 1
        released = False

        def release() -> None:
            nonlocal released, viewers
            if not released:
                released = True
                viewers -= 1

        async def frames() -> AsyncIterator[bytes]:
            current = frame
            last_key = None
            deadline = time.monotonic() + 600
            try:
                while current is not None and time.monotonic() < deadline:
                    if await request.is_disconnected():
                        return
                    key = (current["sessionId"], current["sequenceNumber"])
                    if key != last_key:
                        yield (json.dumps(current, separators=(",", ":")) + "\n").encode()
                        last_key = key
                    await asyncio.sleep(0.125)
                    current = hub.latest() if capture.observer_available else None
            finally:
                release()

        return StreamingResponse(
            frames(),
            media_type="application/x-ndjson",
            headers=_HEADERS,
            background=BackgroundTask(release),
        )

    return router
