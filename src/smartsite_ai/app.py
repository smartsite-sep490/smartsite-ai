"""HTTP contract for the API process, independent of future inference workers."""

import asyncio
import hmac
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from smartsite_ai.config import Settings
from smartsite_ai.identity_api import _EnrollmentBuffer
from smartsite_ai.identity_api import router as identity_router
from smartsite_ai.inference.insightface_recognizer import build_face_recognizer
from smartsite_ai.mf06_identity_api import router as mf06_identity_router
from smartsite_ai.preview_snapshot import LatestPreviewHub, MediaPreviewHub
from smartsite_ai.realtime import parse_zone_polygon, stream_realtime


class LiveHealth(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["smartsite-ai"] = "smartsite-ai"


class ReadyHealth(BaseModel):
    status: Literal["ready", "not_ready"]
    service: Literal["smartsite-ai"] = "smartsite-ai"
    scope: Literal["api"] = "api"
    inference_ready: Literal[False] = False


class Capability(BaseModel):
    status: Literal["not_configured", "configured_demo"] = "not_configured"
    provider: str | None = None
    reason: str


class Capabilities(BaseModel):
    service: Literal["smartsite-ai"] = "smartsite-ai"
    api_version: Literal["v1"] = "v1"
    inference_ready: Literal[False] = False
    capabilities: dict[str, Capability]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.api_ready = True
    try:
        yield
    finally:
        await app.state.preview_hub.close()
        await app.state.media_hub.close()
        app.state.api_ready = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    expose_docs = settings.environment != "production"
    app = FastAPI(
        title="SmartSite AI",
        version="0.1.0",
        description="API foundation. Camera ingestion and inference are not configured.",
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_docs else None,
    )
    app.state.api_ready = False
    app.state.identity_service_token = (
        settings.identity_service_token.get_secret_value()
        if settings.identity_service_token is not None
        else None
    )
    app.state.identity_enrollment_buffer = _EnrollmentBuffer()
    app.state.face_recognizer = build_face_recognizer(settings)
    app.state.realtime_lock = asyncio.Lock()

    snapshot_settings = settings.model_copy(update={"backend_ingestion_url": None})
    app.state.media_hub = MediaPreviewHub(settings.realtime_camera_external_id)
    app.state.media_viewers = 0

    async def produce_preview(sink):
        if settings.realtime_live_media_enabled:
            await stream_realtime(sink, snapshot_settings, media_hub=app.state.media_hub)
        else:
            await stream_realtime(sink, snapshot_settings)

    app.state.preview_hub = LatestPreviewHub(
        produce_preview,
        settings.realtime_camera_external_id,
        app.state.realtime_lock,
    )

    @app.get("/v1/realtime/video-stream", tags=["preview"])
    async def media_stream(request: Request, cameraExternalId: str = ""):
        expected = settings.backend_service_token
        authorization = request.headers.getlist("authorization")
        if (
            expected is None
            or len(authorization) != 1
            or not hmac.compare_digest(
                authorization[0].encode(), ("Bearer " + expected.get_secret_value()).encode()
            )
        ):
            raise HTTPException(401, "Preview source is not authorized")
        if list(request.query_params.multi_items()) != [("cameraExternalId", cameraExternalId)]:
            raise HTTPException(400, "Invalid preview request")
        if cameraExternalId != settings.realtime_camera_external_id:
            raise HTTPException(404, "Preview source is unavailable")
        # A viewer observes an existing producer only; it never starts/restarts capture.
        frame = app.state.media_hub.latest() if settings.realtime_live_media_enabled else None
        if frame is None:
            raise HTTPException(503, "Preview source is unavailable")
        if app.state.media_viewers >= 2:
            raise HTTPException(429, "Preview source is busy", headers={"Retry-After": "2"})
        app.state.media_viewers += 1
        released = False

        def release_viewer():
            nonlocal released
            if not released:
                released = True
                app.state.media_viewers -= 1

        async def frames():
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
                    current = app.state.media_hub.latest()
            finally:
                release_viewer()

        return StreamingResponse(
            frames(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(release_viewer),
        )

    @app.get("/v1/realtime/preview/latest", tags=["preview"])
    async def preview_latest(request: Request, response: Response, cameraExternalId: str = ""):
        expected = settings.backend_service_token
        authorization = request.headers.getlist("authorization")
        if (
            expected is None
            or len(authorization) != 1
            or not hmac.compare_digest(
                authorization[0].encode("utf-8"),
                ("Bearer " + expected.get_secret_value()).encode("utf-8"),
            )
        ):
            raise HTTPException(401, "Preview source is not authorized")
        if list(request.query_params.multi_items()) != [("cameraExternalId", cameraExternalId)]:
            raise HTTPException(400, "Invalid preview request")
        if cameraExternalId != settings.realtime_camera_external_id:
            raise HTTPException(404, "Preview source is unavailable")
        if (
            not settings.realtime_snapshot_enabled
            or not settings.realtime_source
            or not (settings.realtime_model_path or settings.realtime_artifact_spec_path)
        ):
            raise HTTPException(503, "Preview source is unavailable")
        frame = app.state.preview_hub.latest()
        if frame is None:
            raise HTTPException(503, "Preview source is unavailable")
        response.headers["Cache-Control"] = "no-store"
        return {
            "sourceKind": "REPLAY" if Path(settings.realtime_source).is_file() else "LIVE",
            "frame": frame,
        }

    @app.get("/health/live", response_model=LiveHealth, tags=["health"])
    async def live() -> LiveHealth:
        return LiveHealth()

    @app.get(
        "/health/ready",
        response_model=ReadyHealth,
        responses={503: {"model": ReadyHealth, "description": "API lifecycle is not ready"}},
        tags=["health"],
        summary="API process readiness, independent of inference readiness",
    )
    async def ready(response: Response) -> ReadyHealth:
        if not app.state.api_ready:
            response.status_code = 503
            return ReadyHealth(status="not_ready")
        return ReadyHealth(status="ready")

    @app.get("/v1/capabilities", response_model=Capabilities, tags=["capabilities"])
    async def capabilities() -> Capabilities:
        return Capabilities(
            capabilities={
                "camera": Capability(reason="No camera ingestion worker or stream is configured."),
                "detector": Capability(
                    provider="ultralytics-yolo11s + supervision",
                    reason=(
                        "Local realtime demo is opt-in. Production worker is not configured."
                        if settings.realtime_model_path or settings.realtime_artifact_spec_path
                        else "No model weights, PPE model or worker loaded."
                    ),
                ),
                "zone": Capability(
                    provider="supervision",
                    reason="No zones, tracking worker or backend policy contract configured.",
                ),
                "identity": Capability(
                    status=app.state.face_recognizer.capability_status,
                    provider="insightface (candidate)",
                    reason=app.state.face_recognizer.capability_reason,
                ),
                "openai": Capability(
                    provider="openai",
                    reason="Supplementary evidence analysis planned; no API adapter configured.",
                ),
            }
        )

    app.include_router(identity_router)
    app.include_router(mf06_identity_router)

    def _realtime_authorized(websocket: WebSocket) -> bool:
        expected = settings.backend_service_token
        if expected is None:
            return False
        provided = websocket.query_params.get("token", "")
        if not provided:
            return False
        return hmac.compare_digest(provided, expected.get_secret_value())

    async def _reject_realtime(websocket: WebSocket, message: str) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close(code=1011)

    @app.websocket("/ws/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if settings.environment == "production" or app.state.preview_hub.started:
            await _reject_realtime(websocket, "Use the authorized Backend preview")
            return
        if not _realtime_authorized(websocket):
            await _reject_realtime(websocket, "Realtime source is not authorized")
            return
        if app.state.realtime_lock.locked():
            await _reject_realtime(websocket, "Realtime source is already in use")
            return
        if (
            not (settings.realtime_model_path or settings.realtime_artifact_spec_path)
            or not settings.realtime_source
        ):
            await _reject_realtime(
                websocket,
                "Configure a realtime model or artifact spec and SMARTSITE_AI_REALTIME_SOURCE",
            )
            return
        try:
            parse_zone_polygon(settings.realtime_zone_polygon)
        except ValueError:
            await _reject_realtime(websocket, "Realtime zone polygon is invalid")
            return
        async with app.state.realtime_lock:
            await stream_realtime(websocket, settings)

    return app
