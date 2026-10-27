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


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    event_id: str
    schema_version: Literal["1.0.0", "1.1.0"]
    camera_external_id: str
    stream_session_id: str
    captured_at: str
    frame_dimensions: FrameDimensions
    observations: tuple[Observation, ...]

    def _event(self, event_id: str, observations: list[Observation]) -> TechnicalObservationEvent:
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
        limit = next(
            bound.max_length
            for bound in TechnicalObservationEvent.model_fields["observations"].metadata
            if hasattr(bound, "max_length")
        )
        if len(self.observations) <= limit:
            return (self.to_event(),)
        subjects: dict[int, list[Observation]] = {}
        for observation in self.observations:
            subjects.setdefault(observation.track_id, []).append(observation)
        units: list[list[Observation]] = []
        for subject in subjects.values():
            people = [o for o in subject if o.type == "PERSON"]
            if len(people) != 1:
                raise ValueError("Packed subject must have exactly one PERSON context")
            if len(subject) <= limit:
                units.append(subject)
                continue
            context = [o for o in subject if o.type in ("PERSON", "IDENTITY_CANDIDATE")]
            regions: dict[tuple[str, int], list[Observation]] = {}
            for observation in subject:
                if observation.type in ("PPE", "ZONE_ENTRY"):
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
        parts: list[list[Observation]] = []
        for unit in units:
            if not parts or len(parts[-1]) + len(unit) > limit:
                parts.append([])
            parts[-1].extend(unit)
        return tuple(
            self._event(str(uuid5(UUID(self.event_id), f"observation-part:{index}")), part)
            for index, part in enumerate(parts)
        )
