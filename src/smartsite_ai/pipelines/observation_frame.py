"""Internal full-frame observations, distinct from bounded transport events."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid5

from smartsite_ai.domain.observations import (
    FrameDimensions,
    Observation,
    ObservationFramePartition,
    TechnicalObservationEvent,
)
from smartsite_ai.domain.tracked_observations import (
    CameraPpeObservation,
    TrackedZoneEntryObservation,
)
from smartsite_ai.tracking.models import RetiredSegment, SubjectSegment

FrameObservation = Observation | CameraPpeObservation | TrackedZoneEntryObservation


@dataclass(frozen=True, slots=True)
class PartSubjectReference:
    event_id: str
    person_observation_index: int
    subject: SubjectSegment


@dataclass(frozen=True, slots=True)
class TrackedObservationPart:
    """Internal sidecar only; the event wire still contains no continuity claim."""

    event: TechnicalObservationEvent
    subjects: tuple[PartSubjectReference, ...]
    retired_segments: tuple[RetiredSegment, ...]


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    event_id: str
    schema_version: Literal["1.0.0", "1.1.0"]
    camera_external_id: str
    stream_session_id: str
    captured_at: str
    frame_dimensions: FrameDimensions
    observations: tuple[FrameObservation, ...]
    source_sequence_number: int | None = None
    subject_segments: tuple[SubjectSegment, ...] = ()
    retired_segments: tuple[RetiredSegment, ...] = ()
    requires_tracked_wire: bool = False
    camera_configuration_version: int | None = None

    def pack_with_tracking(self, *, sequence_number: int) -> tuple[TrackedObservationPart, ...]:
        """Map internal segments to exact existing part-local PERSON indices.

        This neither serializes continuity nor makes an enqueued/admitted predecessor.
        No-person lifecycle transport requires the future canonical contract, not a
        fabricated empty v1.4 event.
        """
        by_track = self.tracking_subjects(sequence_number=sequence_number)
        return tuple(
            TrackedObservationPart(
                event,
                tuple(
                    PartSubjectReference(event.event_id, index, by_track[observation.track_id])
                    for index, observation in enumerate(event.observations)
                    if observation.type == "PERSON"
                ),
                self.retired_segments,
            )
            for event in self.to_partitioned_events(sequence_number=sequence_number)
        )

    def tracking_subjects(self, *, sequence_number: int) -> dict[int, SubjectSegment]:
        if type(sequence_number) is not int or sequence_number != self.source_sequence_number:
            raise ValueError("Tracking reference sequence must match observed source metadata")
        people = [o.track_id for o in self.observations if o.type == "PERSON"]
        subjects = self.subject_segments
        if (
            not all(isinstance(subject, SubjectSegment) for subject in subjects)
            or len(set(people)) != len(people)
            or len(subjects) != len(people)
            or {subject.track_id for subject in subjects} != set(people)
            or len({subject.segment_id for subject in subjects}) != len(subjects)
        ):
            raise ValueError("Tracking metadata must uniquely match every original PERSON")
        for subject in subjects:
            previous = subject.previous_sequence_number
            if (
                type(subject.track_id) is not int
                or subject.track_id < 1
                or not isinstance(subject.segment_id, UUID)
                or not (
                    (
                        previous is None
                        and subject.reason
                        in {
                            "INITIAL",
                            "RECONNECT",
                            "VIEW_CHANGE",
                            "FRAME_DISCONTINUITY",
                            "PROFILE_CHANGE",
                        }
                    )
                    or (
                        type(previous) is int
                        and 0 <= previous < sequence_number
                        and subject.reason is None
                    )
                )
            ):
                raise ValueError("Invalid internal segment lifecycle claim")
        retired = self.retired_segments
        if (
            not all(isinstance(item, RetiredSegment) for item in retired)
            or len({item.segment_id for item in retired}) != len(retired)
            or {item.segment_id for item in retired} & {s.segment_id for s in subjects}
        ):
            raise ValueError("Retired segments must be distinct from active subjects")
        return {subject.track_id: subject for subject in subjects}

    def _event(self, event_id: str, observations: list[Observation]) -> TechnicalObservationEvent:
        if self.requires_tracked_wire:
            raise ValueError("This frame requires the v1.7 tracked encoder")
        return TechnicalObservationEvent.create(
            event_id=event_id,
            schema_version=self.schema_version,
            camera_external_id=self.camera_external_id,
            stream_session_id=self.stream_session_id,
            captured_at=self.captured_at,
            frame_dimensions=self.frame_dimensions,
            observations=observations,
        )

    def to_event(self) -> TechnicalObservationEvent:
        """Legacy single-event interface; keep wire validation, including its limit."""
        return self._event(self.event_id, list(self.observations))

    def to_partitioned_events(
        self, *, sequence_number: int
    ) -> tuple[TechnicalObservationEvent, ...]:
        """Explicit v1.4 candidate; legacy callers retain their existing wire version."""
        if type(sequence_number) is not int:
            raise ValueError("Source sequence must be an integer")
        parts = self.to_events()
        return tuple(
            TechnicalObservationEvent.create(
                event_id=str(uuid5(UUID(self.event_id), f"observation-v1.4-part:{index}")),
                schema_version="1.4.0",
                camera_external_id=self.camera_external_id,
                stream_session_id=self.stream_session_id,
                captured_at=self.captured_at,
                frame_dimensions=self.frame_dimensions,
                observations=part.observations,
                frame_partition=ObservationFramePartition(
                    frameId=self.event_id,
                    sequenceNumber=str(sequence_number),
                    partIndex=index,
                    partCount=len(parts),
                ),
            )
            for index, part in enumerate(parts)
        )

    def to_events(self) -> tuple[TechnicalObservationEvent, ...]:
        """Pack subjects intact; split oversized subjects only between region groups.

        PERSON/identity context may repeat across parts for one large subject.
        Region findings never repeat, and a region's PPE/entry stays together.
        This is transport packing, not Worker identity or business grouping.
        """
        if self.requires_tracked_wire:
            raise ValueError("This frame requires the v1.7 tracked encoder")
        groups = self._observation_groups()
        if len(groups) == 1:
            return (self.to_event(),)
        return tuple(
            self._event(str(uuid5(UUID(self.event_id), f"observation-part:{index}")), part)
            for index, part in enumerate(groups)
        )

    def tracked_v17_parts(self, *, sequence_number: int) -> tuple[dict[str, object], ...]:
        """Prepare v1.7 partitions before the encoder attaches continuity."""
        if not self.requires_tracked_wire or type(sequence_number) is not int:
            raise ValueError("Expected a tracked v1.7 frame and integer source sequence")
        if sequence_number != self.source_sequence_number:
            raise ValueError("Tracked sequence must match the captured frame")
        groups = self._observation_groups()
        return tuple(
            {
                "eventId": str(uuid5(UUID(self.event_id), f"observation-v1.7-part:{index}")),
                "schemaVersion": "1.7.0",
                "cameraExternalId": self.camera_external_id,
                "streamSessionId": self.stream_session_id,
                "capturedAt": self.captured_at,
                "frameDimensions": self.frame_dimensions.to_wire_dict(),
                "framePartition": ObservationFramePartition(
                    frameId=self.event_id,
                    sequenceNumber=str(sequence_number),
                    partIndex=index,
                    partCount=len(groups),
                ).to_wire_dict(),
                "observations": [observation.to_wire_dict() for observation in observations],
                "evidence": [],
            }
            for index, observations in enumerate(groups)
        )

    def _observation_groups(self) -> list[list[FrameObservation]]:
        limit = next(
            bound.max_length
            for bound in TechnicalObservationEvent.model_fields["observations"].metadata
            if hasattr(bound, "max_length")
        )
        if len(self.observations) <= limit:
            return [list(self.observations)]
        subjects: dict[int, list[FrameObservation]] = {}
        for observation in self.observations:
            subjects.setdefault(observation.track_id, []).append(observation)
        units: list[list[FrameObservation]] = []
        for subject in subjects.values():
            people = [o for o in subject if o.type == "PERSON"]
            if len(people) != 1:
                raise ValueError("Packed subject must have exactly one PERSON context")
            if len(subject) <= limit:
                units.append(subject)
                continue
            context = [o for o in subject if o.type in ("PERSON", "IDENTITY_CANDIDATE")]
            regions: dict[tuple[str, int] | None, list[FrameObservation]] = {}
            for observation in subject:
                if isinstance(observation, CameraPpeObservation):
                    regions.setdefault(None, []).append(observation)
                elif observation.type in ("PPE", "ZONE_ENTRY"):
                    key = (observation.region_id.casefold(), observation.geometry_version)
                    regions.setdefault(key, []).append(observation)
            part = list(context)
            for region in regions.values():
                if len(context) + len(region) > limit:
                    raise ValueError("One subject region exceeds event observation capacity")
                if len(part) + len(region) > limit:
                    units.append(part)
                    part = list(context)
                part.extend(region)
            if len(part) > limit:
                raise ValueError("Subject context exceeds event observation capacity")
            units.append(part)
        parts: list[list[FrameObservation]] = []
        for unit in units:
            if not parts or len(parts[-1]) + len(unit) > limit:
                parts.append([])
            parts[-1].extend(unit)
        return parts
