"""Typed frame envelope capturing frame metadata and payload."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrameEnvelope(BaseModel):
    """Immutable envelope encapsulating ingested video frame and metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    stream_id: str = Field(..., description="Unique identifier for the video stream")
    session_id: str = Field(..., description="Ingestion session identifier")
    camera_external_id: str = Field(..., description="Associated camera external ID")
    captured_at: datetime = Field(..., description="Timezone-aware timestamp of frame capture")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number")
    payload: Any = Field(default=None, description="Raw frame buffer, array, or opaque handle")

    @field_validator("captured_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("captured_at must be timezone-aware")
        return v

    @property
    def captured_at_iso(self) -> str:
        """Return RFC 3339 / ISO 8601 string representation of captured_at."""
        return self.captured_at.isoformat()
