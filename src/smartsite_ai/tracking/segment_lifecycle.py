"""Conservative internal continuity; no wire admission or tracker qualification."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal
from uuid import uuid4

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.tracking.continuity import FrameContinuityPolicy
from smartsite_ai.tracking.models import RetiredSegment, SubjectSegment, TrackedFrame


@dataclass(frozen=True, slots=True)
class SegmentLifecycleState:
    previous: TrackedFrame | None
    view_revision: int | None


class SegmentLifecycle:
    """End segments outside reviewed continuity bounds; consecutive frames by default."""

    def __init__(
        self,
        *,
        max_segment_gap: timedelta | None = None,
        continuity_policy: FrameContinuityPolicy | None = None,
    ) -> None:
        if continuity_policy is not None and not isinstance(
            continuity_policy, FrameContinuityPolicy
        ):
            raise ValueError("Invalid segment continuity policy")
        if max_segment_gap is not None and (
            type(max_segment_gap) is not timedelta or max_segment_gap <= timedelta(0)
        ):
            raise ValueError("max_segment_gap must be a positive timedelta or None")
        self._max_segment_gap = max_segment_gap
        self._continuity_policy = continuity_policy or FrameContinuityPolicy()
        self._previous: TrackedFrame | None = None
        self._view_revision: int | None = None

    def checkpoint_state(self) -> SegmentLifecycleState:
        """Capture reversible continuity state before an uncommitted frame mutates it."""
        return SegmentLifecycleState(previous=self._previous, view_revision=self._view_revision)

    def restore_state(self, state: SegmentLifecycleState) -> None:
        self._previous = state.previous
        self._view_revision = state.view_revision

    def discontinuity_reason(
        self, batch: DetectionBatch, *, view_revision: int | None
    ) -> Literal["RECONNECT", "VIEW_CHANGE", "PROFILE_CHANGE", "FRAME_DISCONTINUITY"] | None:
        """Share the same continuity boundary with new-track admission."""
        previous = self._previous
        scope_changed = previous is not None and (
            previous.batch.stream_id != batch.stream_id
            or previous.batch.session_id != batch.session_id
            or previous.batch.camera_external_id != batch.camera_external_id
        )
        reason = None
        if scope_changed:
            reason = "RECONNECT"
        elif previous is not None:
            prior = previous.batch
            if self._view_revision != view_revision or (prior.frame_width, prior.frame_height) != (
                batch.frame_width,
                batch.frame_height,
            ):
                reason = "VIEW_CHANGE"
            elif (prior.model_artifact_id, prior.model_version, prior.model_sha256) != (
                batch.model_artifact_id,
                batch.model_version,
                batch.model_sha256,
            ):
                reason = "PROFILE_CHANGE"
            elif not self._continuity_policy.allows(prior, batch) or (
                self._max_segment_gap is not None
                and batch.captured_at - prior.captured_at > self._max_segment_gap
            ):
                reason = "FRAME_DISCONTINUITY"
        return reason

    def observe(
        self, frame: TrackedFrame, *, view_revision: int | None, ambiguous_track_ids: set[int]
    ) -> TrackedFrame:
        previous = self._previous
        reason = self.discontinuity_reason(frame.batch, view_revision=view_revision)
        # Never publish old source/session retirements under the new scope.
        old = (
            {}
            if previous is None or reason == "RECONNECT"
            else {subject.track_id: subject for subject in previous.subject_segments}
        )
        current_ids = {person.track_id for person in frame.persons}
        retired = []
        for track_id, subject in old.items():
            if reason is not None or track_id not in current_ids:
                retired.append(
                    RetiredSegment(
                        subject.segment_id,
                        reason
                        if reason is not None
                        else (
                            "AMBIGUOUS_ASSOCIATION"
                            if track_id in ambiguous_track_ids
                            else "LOST_TRACK"
                        ),
                    )
                )
        subjects = []
        for person in frame.persons:
            prior_subject = old.get(person.track_id) if reason is None else None
            subjects.append(
                SubjectSegment(
                    track_id=person.track_id,
                    segment_id=prior_subject.segment_id if prior_subject else uuid4(),
                    previous_sequence_number=previous.batch.sequence_number
                    if prior_subject
                    else None,
                    reason=None if prior_subject else (reason or "INITIAL"),
                )
            )
        result = TrackedFrame(
            frame.batch, frame.persons, tuple(subjects), tuple(retired), frame.provisional_persons
        )
        self._previous = result
        self._view_revision = view_revision
        return result
