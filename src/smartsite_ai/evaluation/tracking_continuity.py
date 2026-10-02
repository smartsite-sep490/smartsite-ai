"""Continuity diagnostics for reviewed matches, without resolving Worker identity.

This module consumes anonymous ground-truth associations already made by a
reviewer. It does not perform detection matching, compute IDF1/HOTA, or establish
that any online correlation or safety policy is correct. Missing ledger frames
are not assumed to contain missed detections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MAX_TRACKING_SAMPLES = 20_000
MAX_TRACKING_IDENTIFIER_LENGTH = 128
MAX_TRACKING_FRAME_INDEX = 2**53 - 1


def _validate_identifier(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_TRACKING_IDENTIFIER_LENGTH
    ):
        raise ValueError("tracking identifier must be a bounded, nonblank, unpadded string")


@dataclass(frozen=True, slots=True)
class TrackingMatch:
    """One reviewed association; person_key is ground truth, never a Worker ID.

    A null track records an explicit unmatched ground-truth person in this frame.
    frame_index refers to the source clip, including across session restarts.
    """

    camera_id: str
    clip_id: str
    session_id: str
    frame_index: int
    person_key: str
    track_id: str | None

    def __post_init__(self) -> None:
        for value in (self.camera_id, self.clip_id, self.session_id, self.person_key):
            _validate_identifier(value)
        if self.track_id is not None:
            _validate_identifier(self.track_id)
        if (
            type(self.frame_index) is not int
            or not 0 <= self.frame_index <= MAX_TRACKING_FRAME_INDEX
        ):
            raise ValueError("frame index must be a bounded nonnegative integer")


@dataclass(frozen=True, slots=True)
class TrackingContinuityReport:
    """Counts from a matched ledger; denominators for accuracy are not inferred."""

    samples: int
    matched_samples: int
    unmatched_samples: int
    distinct_ground_truth_subjects: int
    distinct_predicted_tracks: int
    identity_switches: int
    track_subject_changes: int
    fragments_after_miss: int
    session_boundary_continuations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": "MATCHED_LEDGER_DIAGNOSTICS",
            "samples": self.samples,
            "matchedSamples": self.matched_samples,
            "unmatchedSamples": self.unmatched_samples,
            "distinctGroundTruthSubjects": self.distinct_ground_truth_subjects,
            "distinctPredictedTracks": self.distinct_predicted_tracks,
            "identitySwitches": self.identity_switches,
            "trackSubjectChanges": self.track_subject_changes,
            "fragmentsAfterMiss": self.fragments_after_miss,
            "sessionBoundaryContinuations": self.session_boundary_continuations,
        }


def _validate_matches(matches: Sequence[TrackingMatch]) -> None:
    if len(matches) > MAX_TRACKING_SAMPLES:
        raise ValueError("tracking sample limit exceeded")

    person_frames: set[tuple[str, str, str, int, str]] = set()
    track_frames: set[tuple[str, str, str, int, str]] = set()
    session_ranges: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    for sample in matches:
        if not isinstance(sample, TrackingMatch):
            raise ValueError("tracking samples must be reviewed TrackingMatch records")
        frame = (sample.camera_id, sample.clip_id, sample.session_id, sample.frame_index)
        person_frame = (*frame, sample.person_key)
        if person_frame in person_frames:
            raise ValueError("duplicate or conflicting ground-truth association in a frame")
        person_frames.add(person_frame)
        if sample.track_id is not None:
            track_frame = (*frame, sample.track_id)
            if track_frame in track_frames:
                raise ValueError("predicted track assigned to multiple subjects in a frame")
            track_frames.add(track_frame)
        source_ranges = session_ranges.setdefault((sample.camera_id, sample.clip_id), {})
        start, end = source_ranges.get(sample.session_id, (sample.frame_index, sample.frame_index))
        source_ranges[sample.session_id] = (
            min(start, sample.frame_index),
            max(end, sample.frame_index),
        )

    # A ledger describes one inference run per source clip. Concurrent runs must
    # use separate ledgers; overlapping session ranges cannot imply recovery.
    for source_ranges in session_ranges.values():
        intervals = sorted(source_ranges.values())
        for previous, current in zip(intervals, intervals[1:], strict=False):
            if previous[1] >= current[0]:
                raise ValueError("stream sessions must have nonoverlapping source-frame ranges")


def evaluate_tracking_continuity(
    matches: Sequence[TrackingMatch],
) -> TrackingContinuityReport:
    """Count changes under explicit clip/camera/session and reviewed-person scopes.

    Identity switches retain the last non-null association through explicit
    misses within a session. Fragments require a matched/missed/matched sequence.
    Session-boundary continuations count changes between successive matched
    sessions for a ground-truth person, not proof of online identity recovery.
    """
    _validate_matches(matches)
    ordered = sorted(
        matches,
        key=lambda sample: (
            sample.camera_id,
            sample.clip_id,
            sample.frame_index,
            sample.person_key,
        ),
    )

    subjects: set[tuple[str, str, str]] = set()
    tracks: set[tuple[str, str, str, str]] = set()
    last_person_match: dict[tuple[str, str, str, str], str] = {}
    missed_since_match: set[tuple[str, str, str, str]] = set()
    last_track_subject: dict[tuple[str, str, str, str], str] = {}
    last_matched_session: dict[tuple[str, str, str], str] = {}
    matched_samples = identity_switches = subject_changes = fragments = continuations = 0

    for sample in ordered:
        subject = (sample.camera_id, sample.clip_id, sample.person_key)
        person_session = (*subject, sample.session_id)
        subjects.add(subject)
        if sample.track_id is None:
            if person_session in last_person_match:
                missed_since_match.add(person_session)
            continue

        matched_samples += 1
        previous_track = last_person_match.get(person_session)
        if previous_track is not None and previous_track != sample.track_id:
            identity_switches += 1
        if person_session in missed_since_match:
            fragments += 1
            missed_since_match.remove(person_session)
        last_person_match[person_session] = sample.track_id

        track = (sample.camera_id, sample.clip_id, sample.session_id, sample.track_id)
        tracks.add(track)
        previous_subject = last_track_subject.get(track)
        if previous_subject is not None and previous_subject != sample.person_key:
            subject_changes += 1
        last_track_subject[track] = sample.person_key

        previous_session = last_matched_session.get(subject)
        if previous_session is not None and previous_session != sample.session_id:
            continuations += 1
        last_matched_session[subject] = sample.session_id

    return TrackingContinuityReport(
        samples=len(matches),
        matched_samples=matched_samples,
        unmatched_samples=len(matches) - matched_samples,
        distinct_ground_truth_subjects=len(subjects),
        distinct_predicted_tracks=len(tracks),
        identity_switches=identity_switches,
        track_subject_changes=subject_changes,
        fragments_after_miss=fragments,
        session_boundary_continuations=continuations,
    )
