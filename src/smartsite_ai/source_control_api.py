"""Opt-in source testing API; it neither starts inference nor changes live sources."""

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr

from smartsite_ai.source_activation import SourceActivationError, SourceActivator
from smartsite_ai.source_control_models import SourceDraft, SourcePolicy, SourceRejected
from smartsite_ai.source_resume_models import ResumePrepare


class _ClientGone(Exception):
    pass


async def _connected_probe(
    request: Request, operation: Awaitable[dict[str, object]]
) -> dict[str, object]:
    async def disconnected() -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    work = asyncio.ensure_future(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            work.cancel()
            raise _ClientGone()
        return await work
    finally:
        watcher.cancel()
        if not work.done():
            work.cancel()
        # Probe cleanup terminates/reaps its child before releasing the test lock.
        await asyncio.gather(work, watcher, return_exceptions=True)


class ProbeFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    capturedAt: AwareDatetime
    imageDataUrl: str = Field(
        max_length=350_000, pattern=r"^data:image/jpeg;base64,[A-Za-z0-9+/]+={0,2}$"
    )


def source_control_router(
    *,
    host_id: str,
    label: str,
    scopes: set[tuple[str, str]],
    policy: SourcePolicy,
    service_token: SecretStr,
    probe: Callable[[str, bool, tuple[str, str]], Awaitable[dict[str, object]]],
    activator: SourceActivator | None = None,
) -> APIRouter:
    token = service_token.get_secret_value()
    if not 32 <= len(token) <= 4096 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("A bounded server-only source service token is required")
    expected = ("Bearer " + token).encode("ascii")
    lock = asyncio.Lock()
    router = APIRouter()
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}

    def error(status: int, code: str) -> JSONResponse:
        return JSONResponse({"code": code}, status_code=status, headers=headers)

    def authorize(request: Request, site_id: str, camera_id: str) -> JSONResponse | None:
        auth = request.headers.getlist("authorization")
        if len(auth) != 1 or not secrets.compare_digest(auth[0].encode("utf-8"), expected):
            return error(401, "UNAUTHENTICATED")
        if (site_id, camera_id) not in scopes and (site_id, "*") not in scopes:
            return error(404, "SOURCE_SCOPE_NOT_FOUND")
        return None

    @router.get("/v1/source-control/{site_id}/{camera_id}/capabilities", include_in_schema=False)
    async def capabilities(request: Request, site_id: str, camera_id: str) -> JSONResponse:
        denied = authorize(request, site_id, camera_id)
        if denied is not None:
            return denied
        return JSONResponse(
            {
                "schemaVersion": "1.0.0",
                "hostId": host_id,
                "label": label,
                "sourceKinds": [
                    kind
                    for kind, enabled in (
                        ("WEBCAM", bool(policy.webcams)),
                        ("RTSP", bool(policy.networks)),
                        ("VIDEO", bool(policy.videos)),
                    )
                    if enabled
                ],
                "webcams": [
                    {"id": key, "label": f"USB / Webcam {value}"}
                    for key, value in policy.webcams.items()
                ],
                "videos": [{"id": key, "label": path.name} for key, path in policy.videos.items()],
                "browserCameraSupported": False,
                "applySupported": activator is not None
                and activator.driver.supports((site_id, camera_id)),
                **(
                    {"configurationResumeSupported": True}
                    if activator is not None and activator.supports_resume((site_id, camera_id))
                    else {}
                ),
                "sourceRevision": (activator.registry.read((site_id, camera_id)) or {}).get(
                    "sourceRevision", 0
                )
                if activator
                else 0,
            },
            headers=headers,
        )

    @router.post("/v1/source-control/{site_id}/{camera_id}/probe", include_in_schema=False)
    async def test_source(request: Request, site_id: str, camera_id: str) -> JSONResponse:
        denied = authorize(request, site_id, camera_id)
        if denied is not None:
            return denied
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            return error(400, "INVALID_SOURCE")
        data = bytearray()
        try:
            async with asyncio.timeout(3):
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > 12_000:
                        return error(413, "SOURCE_REQUEST_TOO_LARGE")
        except TimeoutError:
            return error(408, "SOURCE_BODY_TIMEOUT")
        try:
            draft = SourceDraft.model_validate_json(bytes(data))
            source = policy.resolve(draft)
        except (ValueError, SourceRejected):
            return error(400, "SOURCE_NOT_ALLOWED")
        if lock.locked():
            return error(409, "SOURCE_TEST_BUSY")
        try:
            async with lock, asyncio.timeout(12):
                raw = await _connected_probe(
                    request, probe(source, draft.kind != "VIDEO", (site_id, camera_id))
                )
                frame = ProbeFrame.model_validate_json(json.dumps(raw))
                if frame.width * frame.height > 8_000_000:
                    return error(502, "INVALID_SOURCE_PREVIEW")
        except TimeoutError:
            return error(504, "SOURCE_TEST_TIMEOUT")
        except _ClientGone:
            return error(499, "SOURCE_TEST_CANCELLED")
        except asyncio.CancelledError:
            raise
        except Exception:
            return error(503, "SOURCE_UNAVAILABLE")
        receipt = str(uuid4())
        if activator:
            activator.remember(
                (site_id, camera_id), receipt, source, "REPLAY" if draft.kind == "VIDEO" else "LIVE"
            )
        return JSONResponse(
            {
                "schemaVersion": "1.0.0",
                "state": "TESTED",
                "receiptId": receipt,
                "expiresAt": (datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
                "sourceKind": "REPLAY" if draft.kind == "VIDEO" else "LIVE",
                "frame": frame.model_dump(mode="json"),
            },
            headers=headers,
        )

    if activator is not None:

        @router.post(
            "/v1/source-control/{site_id}/{camera_id}/resume-status", include_in_schema=False
        )
        async def resume_status(request: Request, site_id: str, camera_id: str) -> JSONResponse:
            denied = authorize(request, site_id, camera_id)
            if denied is not None:
                return denied
            try:
                body = await activation_body(request)
                if set(body) != {"commandId"} or not isinstance(body["commandId"], str):
                    return error(400, "INVALID_SOURCE_COMMAND")
                command = str(UUID(body["commandId"]))
            except (ValueError, TypeError, KeyError):
                return error(400, "INVALID_SOURCE_COMMAND")
            except TimeoutError:
                return error(408, "SOURCE_BODY_TIMEOUT")
            try:
                async with lock, asyncio.timeout(40):
                    result = await activator.resume_status((site_id, camera_id), command)
                return JSONResponse({"schemaVersion": "1.0.0", **result}, headers=headers)
            except SourceActivationError as exc:
                if str(exc) == "SOURCE_RESUME_NOT_FOUND":
                    return error(404, "SOURCE_RESUME_NOT_FOUND")
                return error(409, "SOURCE_RECONCILIATION_REQUIRED")
            except TimeoutError:
                return error(504, "SOURCE_ACTIVATION_TIMEOUT")
            except Exception:
                return error(503, "SOURCE_RECONCILIATION_REQUIRED")

        @router.post(
            "/v1/source-control/{site_id}/{camera_id}/prepare-resume", include_in_schema=False
        )
        async def prepare_resume(request: Request, site_id: str, camera_id: str) -> JSONResponse:
            denied = authorize(request, site_id, camera_id)
            if denied is not None:
                return denied
            try:
                command = ResumePrepare.model_validate(await activation_body(request))
            except (ValueError, TypeError, KeyError):
                return error(400, "INVALID_SOURCE_COMMAND")
            except TimeoutError:
                return error(408, "SOURCE_BODY_TIMEOUT")
            try:
                async with lock, asyncio.timeout(20):
                    result = await activator.prepare_resume((site_id, camera_id), command)
                return JSONResponse({"schemaVersion": "1.0.0", **result}, headers=headers)
            except SourceActivationError:
                return error(409, "SOURCE_RECONCILIATION_REQUIRED")
            except TimeoutError:
                return error(504, "SOURCE_ACTIVATION_TIMEOUT")
            except Exception:
                return error(503, "SOURCE_RECONCILIATION_REQUIRED")

        @router.post(
            "/v1/source-control/{site_id}/{camera_id}/commit-resume", include_in_schema=False
        )
        async def commit_resume(request: Request, site_id: str, camera_id: str) -> JSONResponse:
            denied = authorize(request, site_id, camera_id)
            if denied is not None:
                return denied
            try:
                body = await activation_body(request)
                if (
                    set(body) != {"commandId", "generation"}
                    or not isinstance(body["commandId"], str)
                    or not isinstance(body["generation"], dict)
                ):
                    return error(400, "INVALID_SOURCE_COMMAND")
                command = str(UUID(body["commandId"]))
                generation = body["generation"]
                if set(generation) != {
                    "workerGenerationId",
                    "streamSessionId",
                    "secret",
                    "expiresAt",
                } or any(not isinstance(value, str) for value in generation.values()):
                    return error(400, "INVALID_SOURCE_COMMAND")
            except (ValueError, TypeError, KeyError):
                return error(400, "INVALID_SOURCE_COMMAND")
            except TimeoutError:
                return error(408, "SOURCE_BODY_TIMEOUT")
            try:
                async with lock, asyncio.timeout(40):
                    result = await activator.commit_resume(
                        (site_id, camera_id), command, generation
                    )
                return JSONResponse({"schemaVersion": "1.0.0", **result}, headers=headers)
            except (SourceActivationError, ValueError, TypeError, KeyError):
                return error(409, "SOURCE_RECONCILIATION_REQUIRED")
            except TimeoutError:
                return error(504, "SOURCE_ACTIVATION_TIMEOUT")
            except Exception:
                return error(503, "SOURCE_RECONCILIATION_REQUIRED")

        @router.post("/v1/source-control/{site_id}/{camera_id}/prepare", include_in_schema=False)
        async def prepare_source(request: Request, site_id: str, camera_id: str) -> JSONResponse:
            denied = authorize(request, site_id, camera_id)
            if denied is not None:
                return denied
            try:
                body = await activation_body(request)
                if set(body) != {"commandId", "receiptId", "expectedRevision"}:
                    return error(400, "INVALID_SOURCE_COMMAND")
                command, receipt = str(UUID(body["commandId"])), str(UUID(body["receiptId"]))
                async with lock, asyncio.timeout(20):
                    result = await activator.prepare(
                        (site_id, camera_id), command, receipt, body["expectedRevision"]
                    )
                return JSONResponse({"schemaVersion": "1.0.0", **result}, headers=headers)
            except (ValueError, TypeError, KeyError):
                return error(409, "SOURCE_ACTIVATION_CONFLICT")
            except TimeoutError:
                return error(504, "SOURCE_ACTIVATION_TIMEOUT")
            except Exception:
                return error(503, "SOURCE_RECONCILIATION_REQUIRED")

        @router.post("/v1/source-control/{site_id}/{camera_id}/commit", include_in_schema=False)
        async def commit_source(request: Request, site_id: str, camera_id: str) -> JSONResponse:
            denied = authorize(request, site_id, camera_id)
            if denied is not None:
                return denied
            try:
                body = await activation_body(request)
                if set(body) != {"commandId", "generation"} or not isinstance(
                    body["generation"], dict
                ):
                    return error(400, "INVALID_SOURCE_COMMAND")
                command = str(UUID(body["commandId"]))
                async with lock, asyncio.timeout(40):
                    result = await activator.commit(
                        (site_id, camera_id), command, body["generation"]
                    )
                return JSONResponse({"schemaVersion": "1.0.0", **result}, headers=headers)
            except (SourceActivationError, ValueError, TypeError, KeyError):
                return error(409, "SOURCE_ACTIVATION_CONFLICT")
            except TimeoutError:
                return error(504, "SOURCE_ACTIVATION_TIMEOUT")
            except Exception:
                return error(503, "SOURCE_RECONCILIATION_REQUIRED")

    return router


async def activation_body(request: Request) -> dict:
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise ValueError()
    raw = bytearray()
    async with asyncio.timeout(3):
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 12_000:
                raise ValueError()
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError()
    return body
