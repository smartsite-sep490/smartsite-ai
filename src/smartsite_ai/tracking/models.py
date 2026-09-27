"""Immutable values exchanged between the tracker and technical pipelines."""

from dataclasses import dataclass

from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    """A person detection associated with a stable identifier for one stream session."""

    track_id: int
    detection: NormalizedDetection

    @property
    def bounding_box(self) -> NormalizedBoundingBox:
        return self.detection.bounding_box


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """Current-frame tracks together with the immutable detector batch they came from."""

    batch: DetectionBatch
    persons: tuple[TrackedPerson, ...]


__all__ = ["TrackedFrame", "TrackedPerson"]
