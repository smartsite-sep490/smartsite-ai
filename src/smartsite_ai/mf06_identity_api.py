"""Dedicated unmounted FastAPI router for MF06 Safety face identity transport."""

import asyncio
import hashlib
import hmac
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError

from smartsite_ai.inference.identity import (
    EncryptedFaceTemplate,
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
)
from smartsite_ai.mf06_identity_wire import (
    MAX_MF06_REQUEST_BODY_BYTES,
    SafetyMf06CandidateWire,
    SafetyMf06InvocationWire,
    SafetyMf06ResolveRequestWire,
    build_mf06_candidate_response,
    build_mf06_unavailable_response,
)

# shortcut: native inference is not preemptible; isolate processes before hard CPU deadlines.
VERIFICATION_TIMEOUT_SECONDS = 8.0
NO_STORE_HEADERS = {"Cache-Control": "no-store"}

router = APIRouter(prefix="/v1/identity/safety-mf06", tags=["identity-safety-mf06"])


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
    candidate_by_hash: dict[str, SafetyMf06CandidateWire],
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
        matched_candidate = candidate_by_hash.get(matched_hash)
        if matched_candidate is None:
            return build_mf06_unavailable_response(invocation, "SERVICE_UNAVAILABLE")

        return build_mf06_candidate_response(invocation, matched_candidate)

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

    if not parsed.candidates:
        return build_mf06_unavailable_response(parsed.invocation, "NO_MATCH")

    invocation_uuid = UUID(parsed.invocation.invocationId)
    candidate_by_hash = {
        candidate.profileReferenceHash: candidate for candidate in parsed.candidates
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

    try:
        result = await asyncio.wait_for(
            recognizer.verify(frame),
            timeout=VERIFICATION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return build_mf06_unavailable_response(parsed.invocation, "ABORTED")
    except asyncio.CancelledError:
        raise
    except Exception:
        return build_mf06_unavailable_response(parsed.invocation, "SERVICE_UNAVAILABLE")

    return _map_verification_result(
        parsed.invocation,
        invocation_uuid,
        candidate_by_hash,
        result,
    )
