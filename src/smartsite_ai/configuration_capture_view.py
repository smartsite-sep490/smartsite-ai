"""Observe an already-running raw capture buffer; never start capture or inference."""

import base64
import hashlib
import secrets
from typing import Any, Literal, Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr


class RawCaptureBuffer(Protocol):
    camera: str

    def latest(self) -> dict[str, Any] | None: ...


def configuration_capture_view_router(
    buffer: RawCaptureBuffer,
    token: SecretStr,
    *,
    source_kind: Literal["LIVE", "REPLAY"],
    enabled: bool,
) -> APIRouter:
    secret = token.get_secret_value()
    if not 32 <= len(secret) <= 4096 or not secret.isascii() or any(c.isspace() for c in secret):
        raise ValueError("Invalid configuration service token")
    if source_kind not in ("LIVE", "REPLAY"):
        raise ValueError("Invalid configuration source kind")
    expected = ("Bearer " + secret).encode("ascii")
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    router = APIRouter()

    @router.get("/v1/realtime/configuration-image", include_in_schema=False)
    async def configuration_image(request: Request) -> JSONResponse:
        def error(status: int) -> JSONResponse:
            return JSONResponse(
                {"code": "CONFIGURATION_IMAGE_UNAVAILABLE"}, status, headers=headers
            )

        authorization = request.headers.getlist("authorization")
        if len(authorization) != 1 or not secrets.compare_digest(
            authorization[0].encode("utf-8"), expected
        ):
            return error(401)
        query = request.query_params.multi_items()
        if len(query) != 1 or query[0][0] != "cameraExternalId" or not query[0][1]:
            return error(400)
        if query[0][1] != buffer.camera:
            return error(404)
        raw = buffer.latest() if enabled else None
        if raw is None or raw.get("type") != "media-frame" or raw.get("mediaVersion") != 1:
            return error(503)
        keys = (
            "cameraExternalId",
            "sessionId",
            "sequenceNumber",
            "capturedAt",
            "width",
            "height",
            "imageDataUrl",
        )
        try:
            if raw["cameraExternalId"] != buffer.camera:
                return error(503)
            frame = {key: raw[key] for key in keys}
            prefix = "data:image/jpeg;base64,"
            image = frame["imageDataUrl"]
            if not isinstance(image, str) or len(image) > 1_400_000 or not image.startswith(prefix):
                return error(503)
            jpeg = base64.b64decode(image[len(prefix) :], validate=True)
            if (
                not 4 <= len(jpeg) <= 1_048_576
                or not jpeg.startswith(b"\xff\xd8")
                or not jpeg.endswith(b"\xff\xd9")
            ):
                return error(503)
            frame["imageSha256"] = hashlib.sha256(jpeg).hexdigest()
        except (KeyError, TypeError, ValueError):
            return error(503)
        return JSONResponse(
            {
                "schemaVersion": 1,
                "purpose": "REGION_CONFIGURATION",
                "sourceKind": source_kind,
                "frame": frame,
            },
            headers=headers,
        )

    return router
