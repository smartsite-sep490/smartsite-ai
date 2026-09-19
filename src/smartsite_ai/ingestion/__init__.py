"""Camera ingestion foundation: FrameEnvelope, FrameSource protocol, queue, and workers."""

from smartsite_ai.ingestion.backoff import ExponentialBackoff
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import BoundedFrameQueue
from smartsite_ai.ingestion.source import (
    FrameSource,
    IngestionError,
    SourceAuthenticationError,
    SourceConnectionError,
    SourceReadError,
    SourceTimeoutError,
    sanitize_stream_url,
)
from smartsite_ai.ingestion.status import (
    StreamMetrics,
    StreamState,
    StreamStatus,
    WorkerStatus,
)
from smartsite_ai.ingestion.worker import (
    CameraIngestionWorker,
    StreamWorker,
)

__all__ = [
    "BoundedFrameQueue",
    "CameraIngestionWorker",
    "ExponentialBackoff",
    "FrameEnvelope",
    "FrameSource",
    "IngestionError",
    "SourceAuthenticationError",
    "SourceConnectionError",
    "SourceReadError",
    "SourceTimeoutError",
    "StreamConfig",
    "StreamMetrics",
    "StreamState",
    "StreamStatus",
    "StreamWorker",
    "WorkerStatus",
    "sanitize_stream_url",
]
