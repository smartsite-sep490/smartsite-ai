"""Typed frame envelope capturing frame metadata and opaque immutable payload."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrameEnvelope(BaseModel):
    """Immutable envelope encapsulating ingested video frame and metadata.

    Enforces strict immutability:
    If a mutable buffer (e.g. bytearray, mutable memoryview) is passed as payload,
    it is copied into an immutable bytes instance to transfer ownership and prevent
    downstream decoders or reused frame buffers from overwriting queued frames.
    """

    model_config = ConfigDict(frozen=True)

    stream_id: str = Field(..., description="Unique identifier for the video stream")
    session_id: str = Field(..., description="Ingestion session identifier")
    camera_external_id: str = Field(..., description="Associated camera external ID")
    captured_at: datetime = Field(..., description="Timezone-aware timestamp of frame capture")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number")
    payload: bytes | str = Field(
        ...,
        description="Opaque immutable frame buffer (bytes) or descriptor handle",
    )

    @field_validator("captured_at")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("captured_at must be timezone-aware")
        return v

    @field_validator("payload", mode="before")
    @classmethod
    def ensure_immutable_payload(cls, v: object) -> bytes | str:
        """Transfer ownership by creating an immutable copy if input is a mutable buffer."""
        if isinstance(v, (bytearray, memoryview)):
            return bytes(v)
        if isinstance(v, (bytes, str)):
            return v
        raise TypeError(f"payload must be bytes, str, or buffer object, got {type(v).__name__}")

    @property
    def captured_at_iso(self) -> str:
        """Return RFC 3339 / ISO 8601 string representation of captured_at."""
        return self.captured_at.isoformat()

    @property
    def buffer(self) -> memoryview:
        """Return a read-only memoryview over payload bytes."""
        if isinstance(self.payload, bytes):
            return memoryview(self.payload)
        return memoryview(self.payload.encode("utf-8"))
