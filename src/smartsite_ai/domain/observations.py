from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)
    coordinate_space: Literal["NORMALIZED_0_1"] = Field(
        default="NORMALIZED_0_1", alias="coordinateSpace"
    )

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 >= self.x2:
            raise ValueError(f"x1 ({self.x1}) must be strictly less than x2 ({self.x2})")
        if self.y1 >= self.y2:
            raise ValueError(f"y1 ({self.y1}) must be strictly less than y2 ({self.y2})")
        return self


class PersonObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["PERSON"] = "PERSON"
    track_id: int = Field(..., ge=0, alias="trackId")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class PpeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["PPE"] = "PPE"
    track_id: int = Field(..., ge=0, alias="trackId")
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"] = Field(..., alias="ppeItem")
    status: Literal["PRESENT", "MISSING"]
    region_id: UUID = Field(..., alias="regionId")
    geometry_version: int = Field(..., ge=1, alias="geometryVersion")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class ZoneEntryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["ZONE_ENTRY"] = "ZONE_ENTRY"
    track_id: int = Field(..., ge=0, alias="trackId")
    region_id: UUID = Field(..., alias="regionId")
    geometry_version: int = Field(..., ge=1, alias="geometryVersion")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class IdentityCandidateObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["IDENTITY_CANDIDATE"] = "IDENTITY_CANDIDATE"
    track_id: int = Field(..., ge=0, alias="trackId")
    status: Literal["CANDIDATE", "UNKNOWN", "UNAVAILABLE"]
    candidate_worker_id: str | None = Field(default=None, min_length=1, alias="candidateWorkerId")
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0, alias="similarityScore")
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0, alias="qualityScore")

    @model_validator(mode="after")
    def validate_status_semantics(self) -> "IdentityCandidateObservation":
        if self.status == "CANDIDATE":
            if self.candidate_worker_id is None:
                raise ValueError("candidateWorkerId is required when status is CANDIDATE")
            if self.similarity_score is None:
                raise ValueError("similarityScore is required when status is CANDIDATE")
        else:
            if self.candidate_worker_id is not None:
                raise ValueError(f"candidateWorkerId is not allowed when status is {self.status}")
            if self.similarity_score is not None:
                raise ValueError(f"similarityScore is not allowed when status is {self.status}")
        return self


Observation = Annotated[
    PersonObservation | PpeObservation | ZoneEntryObservation | IdentityCandidateObservation,
    Field(discriminator="type"),
]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["FRAME", "CROP", "SNAPSHOT"]
    uri: str = Field(..., min_length=1)
    track_id: int | None = Field(default=None, ge=0, alias="trackId")
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class TechnicalObservationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: UUID = Field(..., alias="eventId")
    schema_version: Literal["1.0.0"] = Field(default="1.0.0", alias="schemaVersion")
    camera_external_id: str = Field(..., min_length=1, max_length=128, alias="cameraExternalId")
    stream_session_id: UUID = Field(..., alias="streamSessionId")
    captured_at: str = Field(..., alias="capturedAt")
    frame_dimensions: FrameDimensions = Field(..., alias="frameDimensions")
    observations: list[Observation] = Field(..., min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    def to_wire_dict(self) -> dict[str, Any]:
        """Serializes event to canonical wire dictionary with camelCase keys and no nulls."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def compute_payload_hash(self) -> str:
        """Computes the RFC 8785 canonical SHA-256 hash of this event's wire representation."""
        from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash

        return compute_canonical_payload_hash(self.to_wire_dict())
