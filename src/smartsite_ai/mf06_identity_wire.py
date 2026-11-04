"""Strict wire contracts and resource guards for MF06 Safety face transport."""

import base64
import binascii
import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Literal
from uuid import UUID

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

MAX_MF06_REQUEST_BODY_BYTES = 3 * 1024 * 1024
MAX_MF06_CROP_BYTES = 1_048_576
MAX_MF06_CROP_BASE64_CHARS = 1_400_000
MAX_MF06_CROP_PIXELS = 16_000_000
MAX_MF06_CANDIDATES = 32
EXIF_ORIENTATION_TAG = 0x0112

UUID_WIRE_RE = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"|00000000-0000-0000-0000-000000000000"
    r"|[fF]{8}-[fF]{4}-[fF]{4}-[fF]{4}-[fF]{12})$"
)
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
CAPTURED_AT_UTC_MS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
ORIGINAL_CAPTURED_AT_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:)(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:?\d{2})$"
)
FERNET_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{100,32766}={0,2}$")

UnavailableReasonWire = Literal["NO_MATCH", "UNSUPPORTED", "SERVICE_UNAVAILABLE", "ABORTED"]


def _parse_utc_ms_timestamp(value: str) -> datetime:
    if not CAPTURED_AT_UTC_MS_RE.fullmatch(value):
        raise ValueError("capturedAt must be an ISO-8601 UTC millisecond timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError("capturedAt is not a valid calendar timestamp") from exc


def _normalize_original_captured_at(value: str) -> str:
    if len(value) > 64:
        raise ValueError("originalCapturedAt exceeds maximum length")
    match = ORIGINAL_CAPTURED_AT_RE.fullmatch(value)
    if match is None:
        raise ValueError("originalCapturedAt has invalid ISO-8601 format")

    prefix, seconds_str, frac_str, tz_str = match.groups()
    normalized_seconds = "59" if seconds_str == "60" else seconds_str
    milliseconds = int((frac_str + "000")[:3]) if frac_str else 0

    if tz_str == "Z":
        tz_info = UTC
    else:
        sign = -1 if tz_str[0] == "-" else 1
        raw_offset = tz_str[1:].replace(":", "")
        offset_hours = int(raw_offset[:2])
        offset_minutes = int(raw_offset[2:4])
        if offset_hours > 23 or offset_minutes > 59:
            raise ValueError("originalCapturedAt has invalid timezone offset")
        tz_info = timezone(timedelta(minutes=sign * (offset_hours * 60 + offset_minutes)))

    try:
        base_dt = datetime.fromisoformat(f"{prefix}{normalized_seconds}").replace(tzinfo=tz_info)
        utc_dt = base_dt.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError("originalCapturedAt is not a valid calendar timestamp") from exc

    return f"{utc_dt:%Y-%m-%dT%H:%M:%S}.{milliseconds:03d}Z"


def _validate_uuid_wire(value: str) -> str:
    if not UUID_WIRE_RE.fullmatch(value):
        raise ValueError("Invalid UUID format")
    UUID(value)
    return value


def _validate_jpeg_crop_bytes(crop_bytes: bytes, expected_sha256: str) -> None:
    if len(crop_bytes) < 4 or len(crop_bytes) > MAX_MF06_CROP_BYTES:
        raise ValueError("Crop JPEG byte length is out of bounds")
    if crop_bytes[:3] != b"\xff\xd8\xff" or crop_bytes[-2:] != b"\xff\xd9":
        raise ValueError("Crop bytes are missing JPEG SOI/EOI markers")
    if hashlib.sha256(crop_bytes).hexdigest() != expected_sha256:
        raise ValueError("Crop SHA-256 does not match invocation.cropSha256")

    try:
        with Image.open(BytesIO(crop_bytes)) as image:
            if image.format != "JPEG":
                raise ValueError("Crop image format is not JPEG")
            width, height = image.size
            if width <= 0 or height <= 0 or (width * height) > MAX_MF06_CROP_PIXELS:
                raise ValueError("Crop JPEG dimensions are out of bounds")
            exif = image.getexif()
            orientation = exif.get(EXIF_ORIENTATION_TAG)
            if orientation not in (None, 1):
                raise ValueError("Crop JPEG orientation must be absent or normal (1)")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid JPEG header or unsupported image parameters") from exc


class SafetyMf06CandidateWire(BaseModel):
    """Single encrypted face template candidate scoped to one worker profile revision."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    profileReferenceHash: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_RE)
    profileRevision: str
    encryptedTemplate: str = Field(
        min_length=100,
        max_length=32768,
        pattern=FERNET_TOKEN_RE,
        repr=False,
    )

    @field_validator("profileRevision")
    @classmethod
    def _check_profile_revision(cls, value: str) -> str:
        return _validate_uuid_wire(value)


class SafetyMf06InvocationWire(BaseModel):
    """Strict 25-field correlation and digest contract matching Backend Safety port."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    protocolVersion: Literal["SAFETY_MF06_FACE_V1"]
    purpose: Literal["SAFETY_MF06"]
    invocationId: str
    bindingDigest: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_RE)
    requestId: str
    eventId: str
    payloadHash: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_RE)
    siteId: str
    cameraId: str
    cameraExternalId: str = Field(min_length=1, max_length=128)
    streamSessionId: str
    capturedAt: str
    originalCapturedAt: str = Field(min_length=1, max_length=64)
    personObservationIndex: StrictInt = Field(ge=0, le=255)
    zoneObservationIndex: StrictInt = Field(ge=0, le=255)
    trackId: StrictInt = Field(ge=0, le=9007199254740991)
    regionId: str
    geometryVersion: StrictInt = Field(ge=1, le=9007199254740991)
    zoneId: str
    requestRevision: StrictInt = Field(ge=1, le=2147483647)
    requestedPurpose: Literal["INITIAL_OBSERVATION_ASSESSMENT", "RETROSPECTIVE_REVIEW"]
    frameSha256: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_RE)
    cropSha256: str = Field(min_length=64, max_length=64, pattern=SHA256_HEX_RE)
    cropAlgorithmVersion: Literal["PERSON_FLOOR_CEIL_JPEG95_444_V1"]
    modelVersion: str = Field(min_length=1, max_length=128)

    @field_validator(
        "invocationId",
        "requestId",
        "eventId",
        "siteId",
        "cameraId",
        "streamSessionId",
        "regionId",
        "zoneId",
    )
    @classmethod
    def _check_uuid_fields(cls, value: str) -> str:
        return _validate_uuid_wire(value)

    @field_validator("modelVersion")
    @classmethod
    def _check_model_version(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("modelVersion must not contain leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def _check_cross_field_invariants(self) -> "SafetyMf06InvocationWire":
        if self.personObservationIndex == self.zoneObservationIndex:
            raise ValueError("personObservationIndex and zoneObservationIndex must differ")
        _parse_utc_ms_timestamp(self.capturedAt)
        normalized_original = _normalize_original_captured_at(self.originalCapturedAt)
        if normalized_original != self.capturedAt:
            raise ValueError("originalCapturedAt does not normalize to capturedAt")
        return self

    def parsed_captured_at(self) -> datetime:
        return _parse_utc_ms_timestamp(self.capturedAt)

    def to_echo_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SafetyMf06ResolveRequestWire(BaseModel):
    """Top-level request body for POST /v1/identity/safety-mf06/resolve."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    invocation: SafetyMf06InvocationWire
    cropJpegBase64: str = Field(
        min_length=1,
        max_length=MAX_MF06_CROP_BASE64_CHARS,
        repr=False,
    )
    candidates: list[SafetyMf06CandidateWire] = Field(
        max_length=MAX_MF06_CANDIDATES,
        repr=False,
    )

    @field_validator("candidates")
    @classmethod
    def _check_candidate_uniqueness(
        cls, candidates: list[SafetyMf06CandidateWire]
    ) -> list[SafetyMf06CandidateWire]:
        seen_hashes: set[str] = set()
        seen_revisions: set[str] = set()
        for candidate in candidates:
            if candidate.profileReferenceHash in seen_hashes:
                raise ValueError("Duplicate candidate profileReferenceHash")
            normalized_revision = candidate.profileRevision.lower()
            if normalized_revision in seen_revisions:
                raise ValueError("Duplicate candidate profileRevision")
            seen_hashes.add(candidate.profileReferenceHash)
            seen_revisions.add(normalized_revision)
        return candidates

    def decode_and_validate_crop_bytes(self) -> bytes:
        try:
            crop_bytes = base64.b64decode(self.cropJpegBase64.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("cropJpegBase64 is not valid base64") from exc
        _validate_jpeg_crop_bytes(crop_bytes, self.invocation.cropSha256)
        return crop_bytes


def build_mf06_candidate_response(
    invocation: SafetyMf06InvocationWire,
    candidate: SafetyMf06CandidateWire,
) -> dict[str, Any]:
    return {
        **invocation.to_echo_dict(),
        "state": "CANDIDATE",
        "provenance": "TRUSTED_SERVICE",
        "profileReferenceHash": candidate.profileReferenceHash,
        "profileRevision": candidate.profileRevision,
    }


def build_mf06_unavailable_response(
    invocation: SafetyMf06InvocationWire,
    reason: UnavailableReasonWire,
) -> dict[str, Any]:
    return {
        **invocation.to_echo_dict(),
        "state": "UNAVAILABLE",
        "provenance": "TRUSTED_SERVICE",
        "reason": reason,
    }
