"""Fail-closed, provider-neutral boundary for one-frame face verification.

This module deliberately has no InsightFace import or model loading side effect.
The model, license, enrollment storage, and threshold policy are separate
deployment decisions; callers receive an explicit unavailable result until they
are configured and evaluated.
"""

from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class FaceVerificationFrame(_StrictFrozenModel):
    """An ephemeral JPEG frame. It must not be persisted or logged by an adapter."""

    verification_id: UUID
    captured_at: datetime
    mime_type: Literal["image/jpeg"]
    content: bytes = Field(min_length=1, max_length=5 * 1024 * 1024, exclude=True, repr=False)

    @field_validator("captured_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(UTC)


FaceVerificationStatus = Literal[
    "MATCHED", "UNKNOWN", "LOW_CONFIDENCE", "QUALITY_FAILED", "AI_UNAVAILABLE"
]
FaceScoreBand = Literal["LOW", "MEDIUM", "HIGH"]


class FaceVerificationResult(_StrictFrozenModel):
    """Technical evidence only; it contains neither a worker ID nor an access decision."""

    verification_id: UUID
    status: FaceVerificationStatus
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_profile_reference: str | None = Field(default=None, min_length=1, max_length=128)
    score_band: FaceScoreBand | None = None
    reason_code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9_]+$")

    @model_validator(mode="after")
    def validate_match_fields(self) -> "FaceVerificationResult":
        matched = self.status == "MATCHED"
        if matched and (self.candidate_profile_reference is None or self.score_band is None):
            raise ValueError("MATCHED requires candidate_profile_reference and score_band")
        if not matched and (
            self.candidate_profile_reference is not None or self.score_band is not None
        ):
            raise ValueError("non-matching results must not carry a candidate or score band")
        return self


@runtime_checkable
class FaceRecognizerProtocol(Protocol):
    """One-frame verification against private enrollment templates."""

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        """Return technical evidence without authorization semantics."""
        ...


class UnavailableFaceRecognizer:
    """Default adapter while no reviewed model and private enrollment store exist."""

    async def verify(self, frame: FaceVerificationFrame) -> FaceVerificationResult:
        return FaceVerificationResult(
            verification_id=frame.verification_id,
            status="AI_UNAVAILABLE",
            reason_code="FACE_MODEL_NOT_CONFIGURED",
        )
