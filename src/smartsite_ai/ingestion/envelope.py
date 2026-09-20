"""Immutable, packed BGR24 frames at the ingestion consumer boundary."""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrameEnvelope(BaseModel):
    """A packed frame whose immutable payload owns its storage."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stream_id: str = Field(min_length=1, max_length=128)
    session_id: UUID
    camera_external_id: str = Field(min_length=1, max_length=128)
    captured_at: datetime
    width: int = Field(ge=1, le=16_384)
    height: int = Field(ge=1, le=16_384)
    sequence_number: int = Field(ge=0, le=9_223_372_036_854_775_807)
    pixel_format: Literal["BGR24"] = "BGR24"
    payload: bytes

    @field_validator("captured_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("captured_at must be timezone-aware")
        return v.astimezone(UTC)

    @field_validator("payload", mode="before")
    @classmethod
    def ensure_immutable_payload(cls, v: object) -> object:
        """Copy reusable source buffers before transferring a frame to consumers."""
        if isinstance(v, (bytearray, memoryview)):
            return bytes(v)
        if isinstance(v, str):
            raise ValueError("payload must be bytes, not str")
        return v

    @model_validator(mode="after")
    def validate_packed_payload(self) -> Self:
        if len(self.payload) != self.width * self.height * 3:
            raise ValueError("payload length must equal width * height * 3 for BGR24")
        return self

    @property
    def captured_at_iso(self) -> str:
        """Return the UTC RFC 3339 / ISO 8601 capture timestamp."""
        return self.captured_at.isoformat()

    @property
    def buffer(self) -> memoryview:
        """Return a read-only memoryview over packed BGR24 bytes."""
        return memoryview(self.payload)
