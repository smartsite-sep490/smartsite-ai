"""Mountable observer-only configuration route; no application or worker startup."""

import secrets

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from smartsite_ai.ingestion.configuration_capture import ConfigurationCapture

_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def configuration_image_router(
    capture: ConfigurationCapture, service_token: SecretStr
) -> APIRouter:
    """The explicit runtime owner mounts this router; requests only observe it.

    This does not change the existing inference app or its lifecycle. No capture
    factory, source URL, model, Region or Backend ingestion is needed here.
    """
    token = service_token.get_secret_value()
    if not 32 <= len(token) <= 4096 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("A bounded server-only configuration service token is required")
    expected = f"Bearer {token}".encode("ascii")
    router = APIRouter()

    def error(status: int, code: str) -> JSONResponse:
        headers = {**_HEADERS}
        if status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse({"code": code}, status_code=status, headers=headers)

    @router.get("/v1/realtime/configuration-image", include_in_schema=False)
    async def configuration_image(request: Request) -> JSONResponse:
        authorization = request.headers.getlist("authorization")
        if len(authorization) != 1 or not secrets.compare_digest(
            authorization[0].encode("utf-8"), expected
        ):
            return error(401, "UNAUTHENTICATED")
        query = request.query_params.multi_items()
        if len(query) != 1 or query[0][0] != "cameraExternalId" or not query[0][1]:
            return error(400, "INVALID_CONFIGURATION_QUERY")
        if query[0][1] != capture.camera_external_id:
            return error(404, "CAMERA_NOT_FOUND")
        snapshot = capture.latest()
        if snapshot is None:
            return error(503, "CONFIGURATION_IMAGE_UNAVAILABLE")
        return JSONResponse(snapshot, headers=_HEADERS)

    return router
