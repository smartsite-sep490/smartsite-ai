"""Small deterministic IoU tracker for the technical pipeline boundary."""

from collections import Counter
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson

_DEFAULT_PERSON_CLASS_NAMES: Final[frozenset[str]] = frozenset({"person"})


def bounding_box_iou(first: NormalizedBoundingBox, second: NormalizedBoundingBox) -> float:
    """Return intersection-over-union for two normalized axis-aligned boxes."""

    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0

    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (first_area + second_area - intersection)


@dataclass(slots=True)
class _MutableTrack:
    track_id: int
    detection: NormalizedDetection
    missed_frames: int = 0


class IoUPersonTracker:
    """Track person detections with unambiguous one-to-one IoU association.

    This tracker deliberately tracks only temporal continuity. Its integer track ID
    is not a worker identity and is reset when the stream session changes.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 2,
        person_class_names: frozenset[str] | set[str] = _DEFAULT_PERSON_CLASS_NAMES,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        normalized_names = frozenset(name.casefold() for name in person_class_names)
        if not normalized_names:
            raise ValueError("person_class_names must not be empty")

        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._person_class_names = normalized_names
        self._source_key: tuple[str, UUID] | None = None
        self._camera_external_id: str | None = None
        self._last_sequence: int | None = None
        self._next_track_id = 1
        self._tracks: dict[int, _MutableTrack] = {}

    def update(self, batch: DetectionBatch) -> TrackedFrame:
        """Associate the person detections in ``batch`` with existing tracks."""

        source_key = (batch.stream_id, batch.session_id)
        if self._source_key != source_key:
            self._reset_for_source(source_key)
            self._camera_external_id = batch.camera_external_id
        elif batch.camera_external_id != self._camera_external_id:
            raise ValueError("camera cannot change within a stream session")
        elif self._last_sequence is not None and batch.sequence_number <= self._last_sequence:
            raise ValueError(
                "Detection batches must be supplied in strictly increasing sequence order"
            )

        person_detections = tuple(
            detection
            for detection in batch.detections
            if detection.class_name.casefold() in self._person_class_names
        )
        assignments, retired_track_ids = self._match(person_detections)
        for track_id in retired_track_ids:
            del self._tracks[track_id]
        matched_detection_indexes = set(assignments.values())

        for track_id, track in tuple(self._tracks.items()):
            detection_index = assignments.get(track_id)
            if detection_index is None:
                track.missed_frames += 1
                if track.missed_frames > self._max_missed_frames:
                    del self._tracks[track_id]
                continue

            track.detection = person_detections[detection_index]
            track.missed_frames = 0

        new_track_ids: set[int] = set()
        for detection_index, detection in enumerate(person_detections):
            if detection_index in matched_detection_indexes:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _MutableTrack(track_id=track_id, detection=detection)
            new_track_ids.add(track_id)

        self._last_sequence = batch.sequence_number
        current_track_ids = set(assignments) | new_track_ids
        current_persons = tuple(
            TrackedPerson(track_id=track_id, detection=self._tracks[track_id].detection)
            for track_id in sorted(current_track_ids)
        )
        return TrackedFrame(batch=batch, persons=current_persons)

    def _match(
        self, detections: tuple[NormalizedDetection, ...]
    ) -> tuple[dict[int, int], set[int]]:
        candidates: list[tuple[int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                iou = bounding_box_iou(track.detection.bounding_box, detection.bounding_box)
                # Zero overlap provides no continuity, even at the lowest threshold.
                if iou > 0 and iou >= self._iou_threshold:
                    candidates.append((track_id, detection_index))

        track_counts = Counter(track_id for track_id, _ in candidates)
        detection_counts = Counter(index for _, index in candidates)
        assignments = {
            track_id: index
            for track_id, index in candidates
            if track_counts[track_id] == 1 and detection_counts[index] == 1
        }
        # Retire ambiguous owners permanently: a later frame must not resurrect
        # their pending PPE/Zone state after a crossing, merge or split.
        retired_track_ids = set(track_counts) - assignments.keys()
        return assignments, retired_track_ids

    def _reset_for_source(self, source_key: tuple[str, UUID]) -> None:
        self._source_key = source_key
        self._last_sequence = None
        self._next_track_id = 1
        self._tracks.clear()


__all__ = ["IoUPersonTracker", "bounding_box_iou"]
