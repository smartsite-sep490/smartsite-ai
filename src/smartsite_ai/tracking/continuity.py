"""Explicit technical continuity bounds; raw captured sequence IDs stay unchanged."""

from dataclasses import dataclass
from datetime import timedelta

from smartsite_ai.inference.models import DetectionBatch


@dataclass(frozen=True, slots=True)
class FrameContinuityPolicy:
    max_source_sequence_gap: int = 1
    max_capture_gap: timedelta | None = None

    def __post_init__(self) -> None:
        if (
            type(self.max_source_sequence_gap) is not int
            or not 1 <= self.max_source_sequence_gap <= 64
        ):
            raise ValueError("Source sequence gap must be an integer in [1, 64]")
        if self.max_capture_gap is not None and (
            type(self.max_capture_gap) is not timedelta
            or not timedelta(0) < self.max_capture_gap <= timedelta(seconds=2)
        ):
            raise ValueError("Sampled capture gap must be positive and at most two seconds")
        if self.max_source_sequence_gap > 1 and self.max_capture_gap is None:
            raise ValueError("Sampled continuity requires an explicit capture time bound")

    def allows_sequence(self, previous: int, current: int) -> bool:
        return (
            type(previous) is int
            and type(current) is int
            and 0 <= previous < current <= 2**64 - 1
            and current - previous <= self.max_source_sequence_gap
        )

    def allows(self, previous: DetectionBatch, current: DetectionBatch) -> bool:
        return (
            (previous.stream_id, previous.session_id, previous.camera_external_id)
            == (current.stream_id, current.session_id, current.camera_external_id)
            and (previous.frame_width, previous.frame_height)
            == (current.frame_width, current.frame_height)
            and (previous.model_artifact_id, previous.model_version, previous.model_sha256)
            == (current.model_artifact_id, current.model_version, current.model_sha256)
            and self.allows_sequence(previous.sequence_number, current.sequence_number)
            and current.captured_at > previous.captured_at
            and (
                self.max_capture_gap is None
                or current.captured_at - previous.captured_at <= self.max_capture_gap
            )
        )
