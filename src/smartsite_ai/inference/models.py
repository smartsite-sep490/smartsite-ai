"""Immutable provider-neutral detections normalized to frame coordinates."""

from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import islice
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartsite_ai.ingestion.envelope import FrameEnvelope

_NO_NUL_PATTERN = r"^[^\x00]+$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class NormalizedBoundingBox(_StrictFrozenModel):
    """Axis-aligned bounding box in normalized ``[0, 1]`` coordinates."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    coordinate_space: Literal["NORMALIZED_0_1"] = "NORMALIZED_0_1"

    @model_validator(mode="after")
    def validate_corner_order(self) -> Self:
        if self.x1 >= self.x2:
            raise ValueError("x1 must be strictly less than x2")
        if self.y1 >= self.y2:
            raise ValueError("y1 must be strictly less than y2")
        return self


class NormalizedDetection(_StrictFrozenModel):
    """One detector observation without tracking or business semantics."""

    class_id: int = Field(ge=0, le=2_147_483_647)
    class_name: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: NormalizedBoundingBox


class DetectionBatch(_StrictFrozenModel):
    """Deterministically ordered detections tied to one source frame."""

    stream_id: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    session_id: UUID
    camera_external_id: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    captured_at: datetime
    frame_width: int = Field(ge=1, le=16_384)
    frame_height: int = Field(ge=1, le=16_384)
    sequence_number: int = Field(ge=0, le=9_223_372_036_854_775_807)
    model_artifact_id: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    model_version: str = Field(min_length=1, max_length=64, pattern=_NO_NUL_PATTERN)
    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    detections: tuple[NormalizedDetection, ...] = Field(max_length=1024)

    @field_validator("captured_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("detections")
    @classmethod
    def sort_detections(
        cls, detections: tuple[NormalizedDetection, ...]
    ) -> tuple[NormalizedDetection, ...]:
        return tuple(
            sorted(
                detections,
                key=lambda detection: (
                    -detection.confidence,
                    detection.class_id,
                    detection.bounding_box.x1,
                    detection.bounding_box.y1,
                    detection.bounding_box.x2,
                    detection.bounding_box.y2,
                    detection.class_name,
                ),
            )
        )

    @classmethod
    def from_frame(
        cls,
        frame: FrameEnvelope,
        *,
        model_artifact_id: str,
        model_version: str,
        model_sha256: str,
        detections: Iterable[NormalizedDetection],
    ) -> Self:
        """Build a batch by copying frame identity from the source envelope."""
        return cls(
            stream_id=frame.stream_id,
            session_id=frame.session_id,
            camera_external_id=frame.camera_external_id,
            captured_at=frame.captured_at,
            frame_width=frame.width,
            frame_height=frame.height,
            sequence_number=frame.sequence_number,
            model_artifact_id=model_artifact_id,
            model_version=model_version,
            model_sha256=model_sha256,
            detections=tuple(islice(detections, 1025)),
        )
