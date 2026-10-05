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
    """Current-frame tracks, unique by track ID, with their immutable detector batch.

    This is an internal adapter invariant, not proof of distinct Worker identities.
    Reject malformed output before downstream pipelines can count one track twice
    or advance temporal confirmation more than once in the same frame.
    """

    batch: DetectionBatch
    persons: tuple[TrackedPerson, ...]

    def __post_init__(self) -> None:
        if len({person.track_id for person in self.persons}) != len(self.persons):
            raise ValueError("person track IDs must be unique within a frame")


__all__ = ["TrackedFrame", "TrackedPerson"]
