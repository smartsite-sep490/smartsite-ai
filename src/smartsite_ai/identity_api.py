"""Authenticated, fail-closed HTTP boundary for face enrollment.

Samples live in bounded process memory. Enrollment returns encrypted templates
for Backend PostgreSQL persistence; this service never decides business access.
"""

import asyncio
import base64
import binascii
from datetime import UTC, datetime, timedelta
from hmac import compare_digest
from typing import Final, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smartsite_ai.inference.identity import (
    EncryptedFaceTemplate,
    FaceEnrollmentRequest,
    FaceEnrollmentResult,
    FaceEnrollmentSample,
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
)

_MAX_JPEG_BYTES: Final = 5 * 1024 * 1024
_MAX_ACTIVE_ENROLLMENTS: Final = 128
_SESSION_TTL: Final = timedelta(minutes=10)


class EnrollmentSampleAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted_sample_count: int = Field(ge=1, le=3, serialization_alias="acceptedSampleCount")


class EnrollmentCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    model_version: str | None = Field(default=None, serialization_alias="modelVersion")
    profile_reference: str | None = Field(default=None, serialization_alias="profileReference")
    reason_code: str = Field(serialization_alias="reasonCode")
    encrypted_template: str | None = Field(
        default=None, serialization_alias="encryptedTemplate", repr=False
    )


class DatabaseTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_reference_hash: str = Field(alias="profileReferenceHash", pattern=r"^[0-9a-f]{64}$")
    encrypted_template: str = Field(
        alias="encryptedTemplate", min_length=100, max_length=32768, repr=False
    )


class DatabaseVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jpeg_base64: str = Field(
        alias="jpegBase64", min_length=1, max_length=7 * 1024 * 1024, repr=False
    )
    templates: list[DatabaseTemplate] = Field(max_length=1000, repr=False)
    enrollment_target: Literal["front", "left", "right"] | None = Field(
        default=None, alias="enrollmentTarget"
    )


class VerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    model_version: str | None = Field(default=None, serialization_alias="modelVersion")
    candidate_profile_reference: str | None = Field(
        default=None, serialization_alias="candidateProfileReference"
    )
    score_band: str | None = Field(default=None, serialization_alias="scoreBand")
    reason_code: str = Field(serialization_alias="reasonCode")


class _EnrollmentBuffer:
    def __init__(self) -> None:
        self._entries: dict[UUID, tuple[datetime, dict[int, FaceEnrollmentSample]]] = {}
        self._lock = asyncio.Lock()

    async def add(self, enrollment_id: UUID, sample: FaceEnrollmentSample) -> int:
        async with self._lock:
            self._expire()
            entry = self._entries.get(enrollment_id)
            if entry is None:
                if len(self._entries) >= _MAX_ACTIVE_ENROLLMENTS:
                    raise HTTPException(
                        status_code=503, detail="Identity enrollment capacity unavailable"
                    )
                samples: dict[int, FaceEnrollmentSample] = {}
                self._entries[enrollment_id] = (datetime.now(UTC), samples)
            else:
                _, samples = entry
            expected_index = len(samples) + 1
            if sample.sample_index != expected_index:
                raise HTTPException(
                    status_code=409, detail="Face samples must be submitted in order"
                )
            samples[sample.sample_index] = sample
            return len(samples)

    async def request(self, enrollment_id: UUID) -> FaceEnrollmentRequest:
        async with self._lock:
            self._expire()
            entry = self._entries.get(enrollment_id)
            if entry is None or len(entry[1]) != 3:
                raise HTTPException(status_code=409, detail="Three face samples are required")
            samples = entry[1]
            return FaceEnrollmentRequest(
                enrollment_id=enrollment_id,
                samples=(samples[1], samples[2], samples[3]),
            )

    def _expire(self) -> None:
        cutoff = datetime.now(UTC) - _SESSION_TTL
        self._entries = {
            enrollment_id: entry
            for enrollment_id, entry in self._entries.items()
            if entry[0] >= cutoff
        }


def _authorized(request: Request) -> None:
    configured = request.app.state.identity_service_token
    if configured is None:
        raise HTTPException(status_code=503, detail="Identity enrollment is not configured")
    header = request.headers.get("authorization")
    prefix = "Bearer "
    if header is None or not header.startswith(prefix):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not compare_digest(header[len(prefix) :], configured):
        raise HTTPException(status_code=401, detail="Unauthorized")


router = APIRouter(prefix="/v1/identity", tags=["identity"])


@router.post(
    "/enrollments/{enrollment_id}/samples/{sample_index}",
    response_model=EnrollmentSampleAccepted,
)
async def submit_enrollment_sample(
    enrollment_id: UUID,
    sample_index: int,
    request: Request,
) -> EnrollmentSampleAccepted:
    _authorized(request)
    if sample_index not in (1, 2, 3):
        raise HTTPException(status_code=422, detail="Sample index must be 1, 2, or 3")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
        raise HTTPException(status_code=415, detail="Only JPEG face samples are accepted")
    content = await request.body()
    if not content or len(content) > _MAX_JPEG_BYTES:
        raise HTTPException(status_code=413, detail="Face sample exceeds the size limit")
    sample = FaceEnrollmentSample(
        verification_id=enrollment_id,
        captured_at=datetime.now(UTC),
        mime_type="image/jpeg",
        content=content,
        sample_index=sample_index,
    )
    accepted = await request.app.state.identity_enrollment_buffer.add(enrollment_id, sample)
    return EnrollmentSampleAccepted(accepted_sample_count=accepted)


@router.post(
    "/enrollments/{enrollment_id}/complete",
    response_model=EnrollmentCompletionResponse,
)
async def complete_enrollment(
    enrollment_id: UUID, request: Request
) -> EnrollmentCompletionResponse:
    _authorized(request)
    enrollment = await request.app.state.identity_enrollment_buffer.request(enrollment_id)
    recognizer: FaceRecognizerProtocol = request.app.state.face_recognizer
    result: FaceEnrollmentResult = await recognizer.enroll(enrollment)
    if result.status == "ENROLLED":
        async with request.app.state.identity_enrollment_buffer._lock:
            request.app.state.identity_enrollment_buffer._entries.pop(enrollment_id, None)
    return EnrollmentCompletionResponse(
        status=result.status,
        model_version=result.model_version,
        profile_reference=result.profile_reference,
        reason_code=result.reason_code,
        encrypted_template=result.encrypted_template,
    )


@router.post("/verifications/{verification_id}", response_model=VerificationResponse)
async def verify_face(verification_id: UUID, request: Request) -> VerificationResponse:
    """Compare a transient JPEG against Backend-selected encrypted DB templates."""
    _authorized(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in ("image/jpeg", "application/json"):
        raise HTTPException(
            status_code=415, detail="Only JPEG or database verification is accepted"
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 48 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Face verification exceeds the size limit")
    templates: tuple[EncryptedFaceTemplate, ...] = ()
    enrollment_target = None
    if content_type == "application/json":
        try:
            payload = DatabaseVerification.model_validate_json(body)
            enrollment_target = payload.enrollment_target
            content = base64.b64decode(payload.jpeg_base64, validate=True)
            templates = tuple(
                EncryptedFaceTemplate(
                    profile_reference_hash=item.profile_reference_hash,
                    encrypted_template=item.encrypted_template,
                )
                for item in payload.templates
            )
        except (ValidationError, ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=422, detail="Invalid face verification payload"
            ) from exc
    else:
        content = bytes(body)
    if not content or len(content) > _MAX_JPEG_BYTES:
        raise HTTPException(status_code=413, detail="Face frame exceeds the size limit")
    frame = FaceVerificationFrame(
        verification_id=verification_id,
        captured_at=datetime.now(UTC),
        mime_type="image/jpeg",
        content=content,
        templates=templates,
        enrollment_target=enrollment_target,
    )
    recognizer: FaceRecognizerProtocol = request.app.state.face_recognizer
    result: FaceVerificationResult = await recognizer.verify(frame)
    return VerificationResponse(
        status=result.status,
        model_version=result.model_version,
        candidate_profile_reference=result.candidate_profile_reference,
        score_band=result.score_band,
        reason_code=result.reason_code,
    )
