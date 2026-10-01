"""Camera ingestion foundation: FrameEnvelope, FrameSource protocol, queue, and workers."""

from smartsite_ai.ingestion.backoff import ExponentialBackoff
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.queue import BoundedFrameQueue, QueueClosedError
from smartsite_ai.ingestion.source import (
    FrameIntegrityError,
    FrameSource,
    IngestionError,
    SessionSequenceError,
    SourceAuthenticationError,
    SourceConnectionError,
    SourceReadError,
    SourceTimeoutError,
    classify_error_reason,
    sanitize_message_credentials,
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
    "FrameIntegrityError",
    "FrameSource",
    "IngestionError",
    "OpenCvFrameSource",
    "QueueClosedError",
    "SessionSequenceError",
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
    "classify_error_reason",
    "sanitize_message_credentials",
    "sanitize_stream_url",
]
