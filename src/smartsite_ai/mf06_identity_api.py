"""Bounded MF06 Safety face transport; candidates never establish Worker identity."""

import asyncio
import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from pydantic import ValidationError

from smartsite_ai.inference.identity import (
    EncryptedFaceTemplate,
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
)
from smartsite_ai.mf06_identity_cache import Mf06IdentityCache
from smartsite_ai.mf06_identity_wire import (
    MAX_MF06_REQUEST_BODY_BYTES,
    SafetyMf06InvocationWire,
    SafetyMf06ResolveRequestWire,
    build_mf06_unavailable_response,
)

# shortcut: native inference is not preemptible; isolate processes before hard CPU deadlines.
VERIFICATION_TIMEOUT_SECONDS = 8.0
NO_STORE_HEADERS = {"Cache-Control": "no-store"}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    cache = getattr(app.state, "mf06_face_cache", None)
    if cache is not None:
        cache.close()
    app.state.mf06_face_cache = None
    app.state.mf06_face_draining = False
    try:
        yield
    finally:
        app.state.mf06_face_draining = True
        cache = getattr(app.state, "mf06_face_cache", None)
        if cache is not None:
            cache.close()
        active = getattr(app.state, "mf06_face_active", None)
        if active is not None and not active.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(active, return_exceptions=True)),
                    VERIFICATION_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                raise RuntimeError("MF06_FACE_DRAIN_TIMEOUT") from None


router = APIRouter(
    prefix="/v1/identity/safety-mf06", tags=["identity-safety-mf06"], lifespan=_lifespan
)


def _require_single_bearer_token(request: Request) -> None:
    configured = getattr(request.app.state, "identity_service_token", None)
    if not isinstance(configured, str) or not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Safety face identity is not configured",
            headers=NO_STORE_HEADERS,
        )

    auth_headers = request.headers.getlist("authorization")
    if len(auth_headers) != 1:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers=NO_STORE_HEADERS,
        )

    raw_header = auth_headers[0]
    if not raw_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers=NO_STORE_HEADERS,
        )

    provided_token = raw_header[7:]
    if not provided_token or not hmac.compare_digest(
        provided_token.encode("utf-8"),
        configured.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers=NO_STORE_HEADERS,
        )


def _require_json_content_type(request: Request) -> None:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/json",
            headers=NO_STORE_HEADERS,
        )


async def _read_bounded_body(request: Request, max_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Request body exceeds maximum size",
                    headers=NO_STORE_HEADERS,
                )
        except ValueError:
            pass

    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Request body exceeds maximum size",
                headers=NO_STORE_HEADERS,
            )
        body.extend(chunk)

    if not body:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid MF06 safety face resolve request",
            headers=NO_STORE_HEADERS,
        )
    return bytes(body)


def _map_verification_result(
    invocation: SafetyMf06InvocationWire,
    invocation_uuid: UUID,
    candidate_by_hash: dict[str, str],
    result: Any,
) -> dict[str, Any]:
    if not isinstance(result, FaceVerificationResult):
        return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

    if result.verification_id != invocation_uuid:
        return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

    if result.model_version != invocation.modelVersion:
        return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

    if result.status == "MATCHED":
        raw_reference = result.candidate_profile_reference
        if not isinstance(raw_reference, str) or not raw_reference:
            return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

        matched_hash = hashlib.sha256(raw_reference.encode("utf-8")).hexdigest()
        revision = candidate_by_hash.get(matched_hash)
        if revision is None:
            return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

        return {
            **invocation.to_echo_dict(),
            "state": "CANDIDATE",
            "provenance": "TRUSTED_SERVICE",
            "profileReferenceHash": matched_hash,
            "profileRevision": revision,
        }

    if result.status in ("UNKNOWN", "LOW_CONFIDENCE", "QUALITY_FAILED"):
        return build_mf06_unavailable_response(invocation, "NO_MATCH")

    return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")


@router.post("/resolve")
async def resolve_safety_mf06_face(request: Request, response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    _require_single_bearer_token(request)
    _require_json_content_type(request)

    raw_body = await _read_bounded_body(request, MAX_MF06_REQUEST_BODY_BYTES)
    try:
        parsed = SafetyMf06ResolveRequestWire.model_validate_json(raw_body)
        crop_bytes = parsed.decode_and_validate_crop_bytes()
    except (ValidationError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid MF06 safety face resolve request",
            headers=NO_STORE_HEADERS,
        ) from None

    recognizer: FaceRecognizerProtocol | None = getattr(request.app.state, "face_recognizer", None)
    if recognizer is None:
        return build_mf06_unavailable_response(parsed.invocation, "SERVICE_UNAVAILABLE")

    active = getattr(request.app.state, "mf06_face_active", None)
    if getattr(request.app.state, "mf06_face_draining", False):
        return build_mf06_unavailable_response(parsed.invocation, "SERVICE_UNAVAILABLE")

    invocation_uuid = UUID(parsed.invocation.invocationId)
    invocation_key = str(invocation_uuid)
    fingerprint = hashlib.sha256(
        json.dumps(parsed.model_dump(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache = getattr(request.app.state, "mf06_face_cache", None)
    if cache is None or cache.provider is not recognizer:
        if cache is not None:
            cache.close()
        cache = Mf06IdentityCache(recognizer)
        request.app.state.mf06_face_cache = cache
    try:
        cached = cache.lookup(invocation_key, fingerprint)
    except ValueError:
        raise HTTPException(
            409, "MF06 invocation input conflict", headers=NO_STORE_HEADERS
        ) from None
    if cached is not None:
        return cached
    if active is not None and not active.done():
        prior = getattr(request.app.state, "mf06_face_active_binding", None)
        if (
            prior is not None
            and prior[0] is recognizer
            and prior[1] == invocation_key
            and prior[2] != fingerprint
        ):
            raise HTTPException(409, "MF06 invocation input conflict", headers=NO_STORE_HEADERS)
        return build_mf06_unavailable_response(parsed.invocation, "SERVICE_UNAVAILABLE")

    if not parsed.candidates:
        return build_mf06_unavailable_response(parsed.invocation, "NO_MATCH")

    candidate_by_hash = {
        candidate.profileReferenceHash: candidate.profileRevision for candidate in parsed.candidates
    }
    frame = FaceVerificationFrame(
        verification_id=invocation_uuid,
        captured_at=parsed.invocation.parsed_captured_at(),
        mime_type="image/jpeg",
        content=crop_bytes,
        enrollment_target=None,
        gate_presence_session=None,
        templates=tuple(
            EncryptedFaceTemplate(
                profile_reference_hash=candidate.profileReferenceHash,
                encrypted_template=candidate.encryptedTemplate,
            )
            for candidate in parsed.candidates
        ),
    )

    # No await between slot inspection and reservation on this application's event loop.
    active = asyncio.create_task(
        _verify_and_remember(
            recognizer, frame, parsed.invocation, candidate_by_hash, cache, fingerprint
        )
    )
    request.app.state.mf06_face_active = active
    request.app.state.mf06_face_active_binding = (recognizer, invocation_key, fingerprint)

    def settled(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()  # Observe late failures after the HTTP caller has left.
        if getattr(request.app.state, "mf06_face_active", None) is task:
            request.app.state.mf06_face_active = None
            request.app.state.mf06_face_active_binding = None

    active.add_done_callback(settled)
    try:
        result = await asyncio.wait_for(
            asyncio.shield(active),
            timeout=VERIFICATION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return build_mf06_unavailable_response(parsed.invocation, "ABORTED")
    except asyncio.CancelledError:
        raise
    except Exception:
        return build_mf06_unavailable_response(parsed.invocation, "SERVICE_UNAVAILABLE")

    return result


async def _verify_and_remember(
    recognizer: FaceRecognizerProtocol,
    frame: FaceVerificationFrame,
    invocation: SafetyMf06InvocationWire,
    candidates: dict[str, str],
    cache: Mf06IdentityCache,
    fingerprint: str,
) -> dict[str, Any]:
    result = await recognizer.verify(frame)
    response = _map_verification_result(invocation, frame.verification_id, candidates, result)
    cache.remember(str(frame.verification_id), fingerprint, response)
    return response
