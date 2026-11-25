"""Typed tracked wire structure; claims do not attest continuity or identity."""

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.observations import (
    MAX_SAFE_INTEGER,
    BoundingBox,
    IdentityCandidateObservation,
    JsonInteger,
    ObservationFramePartition,
    PersonObservation,
    PpeObservation,
    StrictWireModel,
    TechnicalObservationEvent,
    ZoneEntryObservation,
    validate_uuid_str,
)

RetirementReason = Literal[
    "RECONNECT",
    "AMBIGUOUS_ASSOCIATION",
    "LOST_TRACK",
    "FRAME_DISCONTINUITY",
    "VIEW_CHANGE",
    "CONFIGURATION_CHANGE",
    "EXPIRED_INTERVAL",
    "PROFILE_CHANGE",
]


class TrackedScopeTransitionRequired(RuntimeError):
    """The trusted owner must finish the prior scope before authorizing a new source."""


class StartLink(StrictWireModel):
    kind: Literal["START"]
    reason: Literal["INITIAL", RetirementReason]


class ContinueLink(StrictWireModel):
    kind: Literal["CONTINUE"]
    previous_event_id: str = Field(alias="previousEventId")
    previous_person_observation_index: JsonInteger = Field(
        alias="previousPersonObservationIndex", ge=0, le=255
    )

    @field_validator("previous_event_id")
    @classmethod
    def canonical_previous_event(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="previousEventId")


class SubjectClaim(StrictWireModel):
    person_observation_index: JsonInteger = Field(alias="personObservationIndex", ge=0, le=255)
    segment_id: str = Field(alias="segmentId")
    link: Annotated[StartLink | ContinueLink, Field(discriminator="kind")]

    @field_validator("segment_id")
    @classmethod
    def canonical_segment(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="segmentId")


class RetiredSegment(StrictWireModel):
    segment_id: str = Field(alias="segmentId")
    reason: RetirementReason

    @field_validator("segment_id")
    @classmethod
    def canonical_segment(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="segmentId")


class CompleteCoverage(StrictWireModel):
    status: Literal["COMPLETE"]


class IncompleteCoverage(StrictWireModel):
    status: Literal["INCOMPLETE"]
    reason: Literal["INFERENCE_ERROR", "AMBIGUOUS_PROCESSING", "FRAME_DROPPED"]


class FrameContinuity(StrictWireModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    profile_sha256: str = Field(alias="profileSha256", pattern=r"^[0-9a-f]{64}$")
    coverage: Annotated[CompleteCoverage | IncompleteCoverage, Field(discriminator="status")]
    subjects: list[SubjectClaim] = Field(max_length=256)
    retired_segments: list[RetiredSegment] = Field(alias="retiredSegments", max_length=256)

    @model_validator(mode="after")
    def incomplete_has_no_claims(self) -> Self:
        if self.coverage.status == "INCOMPLETE" and self.subjects:
            raise ValueError("Incomplete coverage cannot make subject claims")
        return self


class CameraPpeObservation(StrictWireModel):
    type: Literal["PPE"]
    track_id: JsonInteger = Field(alias="trackId", ge=0, le=MAX_SAFE_INTEGER)
    scope: Literal["CAMERA"]
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"] = Field(
        alias="ppeItem"
    )
    status: Literal["PRESENT", "MISSING"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class RegionalSupplementalPpeObservation(StrictWireModel):
    """Existing v1.4 regional semantics; never widen CAMERA or legacy DTOs."""

    type: Literal["PPE"]
    track_id: JsonInteger = Field(alias="trackId", ge=0, le=MAX_SAFE_INTEGER)
    ppe_item: Literal["MASK", "FULL_BODY_HARNESS", "LANYARD_HOOK"] = Field(alias="ppeItem")
    status: Literal["PRESENT", "MISSING"]
    region_id: str = Field(alias="regionId")
    geometry_version: JsonInteger = Field(alias="geometryVersion", ge=1, le=MAX_SAFE_INTEGER)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")

    @field_validator("region_id")
    @classmethod
    def valid_region(cls, value: str) -> str:
        return validate_uuid_str(value, "regionId")

    @model_validator(mode="after")
    def present_only_fall_protection(self) -> Self:
        if self.ppe_item != "MASK" and self.status != "PRESENT":
            raise ValueError("Harness and Hook observations must be PRESENT only")
        return self


class TrackedZoneEntryObservation(ZoneEntryObservation):
    entry_origin: Literal["OUTSIDE_TO_INSIDE", "INSIDE_ENTRY_UNOBSERVED"] = Field(
        alias="entryOrigin"
    )


# The two PPE/Zone shapes intentionally share a type. A plain strict union keeps
# additional fields forbidden without changing legacy discriminated unions.
TrackedObservation = (
    PersonObservation
    | PpeObservation
    | RegionalSupplementalPpeObservation
    | CameraPpeObservation
    | TrackedZoneEntryObservation
    | ZoneEntryObservation
    | IdentityCandidateObservation
)


class TrackedObservationEvent(TechnicalObservationEvent):
    schema_version: Literal["1.5.0", "1.6.0", "1.7.0"] = Field(alias="schemaVersion")
    observations: list[TrackedObservation] = Field(min_length=0, max_length=256)
    frame_partition: ObservationFramePartition = Field(alias="framePartition")
    continuity: FrameContinuity

    @model_validator(mode="after")
    def enforce_ppe_version_boundary(self) -> Self:
        """Replace only the legacy version validator; primitive guards are inherited."""
        people = Counter(item.track_id for item in self.observations if item.type == "PERSON")
        if any(people.get(item.track_id) != 1 for item in self.observations):
            raise ValueError("Each track requires exactly one same-part PERSON")
        if any(item.sha256 is None for item in self.evidence):
            raise ValueError("Tracked evidence requires its original sha256")

        indices: set[int] = set()
        segments: set[str] = set()
        predecessors: set[tuple[str, int]] = set()
        continuing_tracks: set[int] = set()
        for claim in self.continuity.subjects:
            index = claim.person_observation_index
            if index >= len(self.observations) or self.observations[index].type != "PERSON":
                raise ValueError("Subject index must identify an original PERSON")
            if index in indices or claim.segment_id in segments:
                raise ValueError("PERSON and segment claims must be unique")
            indices.add(index)
            segments.add(claim.segment_id)
            if isinstance(claim.link, ContinueLink):
                prior = (claim.link.previous_event_id, claim.link.previous_person_observation_index)
                if prior[0] == self.event_id.lower() or prior in predecessors:
                    raise ValueError("Predecessor cannot be this event or be shared")
                predecessors.add(prior)
                continuing_tracks.add(self.observations[index].track_id)
        complete = self.continuity.coverage.status == "COMPLETE"
        if complete and any(
            item.type == "PERSON" and index not in indices
            for index, item in enumerate(self.observations)
        ):
            raise ValueError("Every original PERSON requires an explicit subject claim")

        retired: set[str] = set()
        for item in self.continuity.retired_segments:
            if item.segment_id in retired or item.segment_id in segments:
                raise ValueError("Retired segments must be unique and not active")
            retired.add(item.segment_id)

        positioned = {
            item.track_id
            for item in self.observations
            if isinstance(item, PersonObservation) and item.bounding_box is not None
        }
        regional = {
            item.track_id
            for item in self.observations
            if isinstance(item, PpeObservation | RegionalSupplementalPpeObservation)
        }
        for item in self.observations:
            if isinstance(item, CameraPpeObservation):
                if self.schema_version != "1.7.0":
                    raise ValueError("Camera PPE requires explicit wire v1.7")
                if item.track_id not in positioned or item.track_id in regional:
                    raise ValueError("Camera PPE requires positioned PERSON and unmixed scope")
            if isinstance(item, ZoneEntryObservation):
                has_origin = isinstance(item, TrackedZoneEntryObservation)
                if has_origin != (self.schema_version != "1.5.0"):
                    raise ValueError("Zone entry origin does not match tracked wire version")
                if (
                    has_origin
                    and item.entry_origin == "OUTSIDE_TO_INSIDE"
                    and (not complete or item.track_id not in continuing_tracks)
                ):
                    raise ValueError("Observed crossing requires complete continuing subject")
        return self
