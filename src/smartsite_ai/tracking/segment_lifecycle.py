"""Conservative internal continuity; no wire admission or tracker qualification."""

from uuid import uuid4

from smartsite_ai.tracking.models import RetiredSegment, SubjectSegment, TrackedFrame


class SegmentLifecycle:
    """A missing processed frame ends a segment, even if a numeric ID survives."""

    def __init__(self) -> None:
        self._previous: TrackedFrame | None = None
        self._view_revision: int | None = None

    def observe(
        self, frame: TrackedFrame, *, view_revision: int | None, ambiguous_track_ids: set[int]
    ) -> TrackedFrame:
        previous = self._previous
        batch = frame.batch
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
            elif (
                batch.sequence_number != prior.sequence_number + 1
                or batch.captured_at <= prior.captured_at
            ):
                reason = "FRAME_DISCONTINUITY"
        # Never publish old source/session retirements under the new scope.
        old = (
            {}
            if previous is None or scope_changed
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
        result = TrackedFrame(frame.batch, frame.persons, tuple(subjects), tuple(retired))
        self._previous = result
        self._view_revision = view_revision
        return result
