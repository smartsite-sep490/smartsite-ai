"""Immutable values exchanged between the tracker and technical pipelines."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

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
class SubjectSegment:
    """Internal association claim, never a qualified identity or event predecessor."""

    track_id: int
    segment_id: UUID
    previous_sequence_number: int | None
    reason: (
        Literal["INITIAL", "RECONNECT", "VIEW_CHANGE", "FRAME_DISCONTINUITY", "PROFILE_CHANGE"]
        | None
    )


@dataclass(frozen=True, slots=True)
class RetiredSegment:
    segment_id: UUID
    reason: Literal[
        "LOST_TRACK",
        "AMBIGUOUS_ASSOCIATION",
        "VIEW_CHANGE",
        "FRAME_DISCONTINUITY",
        "PROFILE_CHANGE",
    ]


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """Current-frame tracks, unique by track ID, with their immutable detector batch.

    This is an internal adapter invariant, not proof of distinct Worker identities.
    Reject malformed output before downstream pipelines can count one track twice
    or advance temporal confirmation more than once in the same frame.
    """

    batch: DetectionBatch
    persons: tuple[TrackedPerson, ...]
    subject_segments: tuple[SubjectSegment, ...] = ()
    retired_segments: tuple[RetiredSegment, ...] = ()

    def __post_init__(self) -> None:
        if len({person.track_id for person in self.persons}) != len(self.persons):
            raise ValueError("person track IDs must be unique within a frame")


__all__ = ["TrackedFrame", "TrackedPerson"]
