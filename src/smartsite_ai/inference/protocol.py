"""Provider-neutral async detector protocol."""

from typing import Protocol, runtime_checkable

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.ingestion.envelope import FrameEnvelope


@runtime_checkable
class DetectorProtocol(Protocol):
    """Consume an immutable frame and return normalized detections."""

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        """Detect objects in one frame without provider-specific semantics."""
        ...
