"""Tracker boundary used by MF05/MF06 pipelines."""

from typing import Protocol, runtime_checkable

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.tracking.models import TrackedFrame


@runtime_checkable
class TrackerProtocol(Protocol):
    """Associate person detections across ordered frames of one stream session."""

    def update(self, batch: DetectionBatch) -> TrackedFrame:
        """Return current-frame person tracks without making an identity decision."""
        ...


__all__ = ["TrackerProtocol"]
