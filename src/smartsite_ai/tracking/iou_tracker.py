"""Small deterministic IoU tracker for the technical pipeline boundary."""

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from typing import Final
from uuid import UUID

from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson
from smartsite_ai.tracking.segment_lifecycle import SegmentLifecycle

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
        max_segment_gap: timedelta | None = None,
        association_iou_margin: float | None = None,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if association_iou_margin is not None and (
            type(association_iou_margin) not in (int, float)
            or not isfinite(association_iou_margin)
            or not 0 < association_iou_margin <= 1
        ):
            raise ValueError("association_iou_margin must be finite in (0, 1] or None")
        normalized_names = frozenset(name.casefold() for name in person_class_names)
        if not normalized_names:
            raise ValueError("person_class_names must not be empty")

        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._association_iou_margin = association_iou_margin
        self._person_class_names = normalized_names
        self._source_key: tuple[str, UUID] | None = None
        self._camera_external_id: str | None = None
        self._last_sequence: int | None = None
        self._view_revision: int | None = None
        self._next_track_id = 1
        self._tracks: dict[int, _MutableTrack] = {}
        self._segments = SegmentLifecycle(max_segment_gap=max_segment_gap)

    def update(self, batch: DetectionBatch, *, view_revision: int | None = None) -> TrackedFrame:
        """Associate the person detections in ``batch`` with existing tracks."""

        if view_revision is not None and (
            type(view_revision) is not int or not 1 <= view_revision <= 9_007_199_254_740_991
        ):
            raise ValueError("view_revision must be a positive safe integer or None")
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

        if self._view_revision != view_revision:
            # Keep the allocation counter: old IDs must not resurrect in this session.
            self._tracks.clear()
        self._view_revision = view_revision

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
        return self._segments.observe(
            TrackedFrame(batch=batch, persons=current_persons),
            view_revision=view_revision,
            ambiguous_track_ids=retired_track_ids,
        )

    def _match(
        self, detections: tuple[NormalizedDetection, ...]
    ) -> tuple[dict[int, int], set[int]]:
        candidates: list[tuple[int, int]] = []
        scores: dict[tuple[int, int], float] = {}
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                iou = bounding_box_iou(track.detection.bounding_box, detection.bounding_box)
                # Zero overlap provides no continuity, even at the lowest threshold.
                if iou > 0 and iou >= self._iou_threshold:
                    candidates.append((track_id, detection_index))
                    if self._association_iou_margin is not None:
                        scores[track_id, detection_index] = iou

        track_counts = Counter(track_id for track_id, _ in candidates)
        detection_counts = Counter(index for _, index in candidates)
        assignments = {
            track_id: index
            for track_id, index in candidates
            if track_counts[track_id] == 1 and detection_counts[index] == 1
        }
        if self._association_iou_margin is not None:
            assignments.update(self._separated_components(scores, self._association_iou_margin))
        # Retire ambiguous owners permanently: a later frame must not resurrect
        # their pending PPE/Zone state after a crossing, merge or split.
        retired_track_ids = set(track_counts) - assignments.keys()
        return assignments, retired_track_ids

    def _separated_components(
        self, scores: dict[tuple[int, int], float], margin: float
    ) -> dict[int, int]:
        """Opt-in geometric evidence only; incomplete or weak components still retire."""
        by_track: dict[int, dict[int, float]] = {}
        by_detection: dict[int, dict[int, float]] = {}
        for (track_id, index), score in scores.items():
            by_track.setdefault(track_id, {})[index] = score
            by_detection.setdefault(index, {})[track_id] = score

        def separated_best(values: dict[int, float]) -> int | None:
            ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) == 1 or ranked[0][1] - ranked[1][1] > margin:
                return ranked[0][0]
            return None

        best_track = {track: separated_best(values) for track, values in by_track.items()}
        best_detection = {index: separated_best(values) for index, values in by_detection.items()}
        remaining = set(by_track)
        accepted: dict[int, int] = {}
        while remaining:
            pending = {min(remaining)}
            tracks: set[int] = set()
            indexes: set[int] = set()
            while pending:
                track = pending.pop()
                if track in tracks:
                    continue
                tracks.add(track)
                for index in by_track[track]:
                    indexes.add(index)
                    pending.update(set(by_detection[index]) - tracks)
            remaining.difference_update(tracks)
            matches = {
                track: index
                for track in tracks
                if (index := best_track[track]) is not None and best_detection[index] == track
            }
            # A high-scoring survivor must not inherit an unresolved merge/split component.
            if len(matches) == len(tracks) == len(indexes):
                accepted.update(matches)
        return accepted

    def _reset_for_source(self, source_key: tuple[str, UUID]) -> None:
        self._source_key = source_key
        self._last_sequence = None
        self._next_track_id = 1
        self._tracks.clear()


__all__ = ["IoUPersonTracker", "bounding_box_iou"]
