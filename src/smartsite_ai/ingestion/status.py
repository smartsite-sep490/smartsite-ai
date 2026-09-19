"""Typed status and health snapshot models for camera streams and worker."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StreamState(StrEnum):
    """Lifecycle state of an individual stream worker."""

    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    BACKOFF = "backoff"
    STOPPED = "stopped"
    ERROR = "error"


class StreamMetrics(BaseModel):
    """Monotonic and operational metrics for an ingestion stream."""

    model_config = ConfigDict(frozen=True)

    frames_enqueued: int = Field(default=0, ge=0)
    frames_dequeued: int = Field(default=0, ge=0)
    frames_dropped: int = Field(default=0, ge=0)
    connection_errors: int = Field(default=0, ge=0)
    read_errors: int = Field(default=0, ge=0)
    reconnect_attempts: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    last_frame_timestamp: datetime | None = None
    last_error: str | None = None
    started_at: datetime | None = None
    connected_at: datetime | None = None
    stopped_at: datetime | None = None


class StreamStatus(BaseModel):
    """Observable status snapshot for a camera stream."""

    model_config = ConfigDict(frozen=True)

    stream_id: str
    camera_external_id: str
    sanitized_url: str
    state: StreamState
    metrics: StreamMetrics
    queue_size: int
    queue_capacity: int


class WorkerStatus(BaseModel):
    """Observable status snapshot for the overall CameraIngestionWorker."""

    model_config = ConfigDict(frozen=True)

    service: Literal["smartsite-ai"] = "smartsite-ai"
    status: Literal["idle", "running", "stopping", "stopped"]
    # Never claim inference_ready: true in camera worker foundation
    inference_ready: Literal[False] = False
    stream_count: int
    active_stream_count: int
    streams: dict[str, StreamStatus]
