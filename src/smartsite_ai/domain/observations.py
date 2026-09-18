import re
from datetime import date
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
RFC3339_DATETIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[tT]"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.\d+)?"
    r"(?:[zZ]|[+-]\d{2}(?::?\d{2})?)$"
)


def validate_uuid_str(v: Any, field_name: str) -> str:
    if not isinstance(v, str) or not UUID_RE.match(v):
        raise ValueError(f"{field_name} must be a valid UUID string")
    try:
        UUID(v)
    except Exception as e:
        raise ValueError(f"Invalid UUID in {field_name}: {e}") from e
    return v


def validate_rfc3339_captured_at(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError("capturedAt must be a string")
    match = RFC3339_DATETIME_RE.match(v)
    if not match:
        raise ValueError(f"capturedAt '{v}' does not match RFC 3339 date-time format")
    year_s, month_s, day_s, hour_s, min_s, sec_s = match.groups()
    year, month, day = int(year_s), int(month_s), int(day_s)
    hour, minute, second = int(hour_s), int(min_s), int(sec_s)

    if hour > 23 or minute > 59:
        raise ValueError(f"Invalid time components in capturedAt '{v}'")
    if second > 60:
        raise ValueError(f"Second component cannot exceed 60 in capturedAt '{v}'")
    if second == 60 and (hour != 23 or minute != 59):
        raise ValueError(f"Leap second only valid at 23:59:60 in capturedAt '{v}'")

    try:
        date(year, month, day)
    except ValueError as e:
        raise ValueError(f"Invalid calendar date in capturedAt '{v}': {e}") from e
    return v


class StrictWireModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        populate_by_name=False,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, val in data.items():
                if val is None:
                    raise ValueError(f"Explicit null is not permitted for property '{key}'")
        return data

    def to_wire_dict(self) -> dict[str, Any]:
        """Serializes model to canonical wire dictionary with camelCase keys and no nulls."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class FrameDimensions(StrictWireModel):
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class BoundingBox(StrictWireModel):
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)
    coordinate_space: Literal["NORMALIZED_0_1"] = Field(..., alias="coordinateSpace")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 >= self.x2:
            raise ValueError(f"x1 ({self.x1}) must be strictly less than x2 ({self.x2})")
        if self.y1 >= self.y2:
            raise ValueError(f"y1 ({self.y1}) must be strictly less than y2 ({self.y2})")
        return self


class PersonObservation(StrictWireModel):
    type: Literal["PERSON"] = Field(..., alias="type")
    track_id: int = Field(..., ge=0, alias="trackId")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class PpeObservation(StrictWireModel):
    type: Literal["PPE"] = Field(..., alias="type")
    track_id: int = Field(..., ge=0, alias="trackId")
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"] = Field(..., alias="ppeItem")
    status: Literal["PRESENT", "MISSING"]
    region_id: str = Field(..., alias="regionId")
    geometry_version: int = Field(..., ge=1, alias="geometryVersion")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")

    @field_validator("region_id")
    @classmethod
    def check_region_id(cls, v: str) -> str:
        return validate_uuid_str(v, "regionId")


class ZoneEntryObservation(StrictWireModel):
    type: Literal["ZONE_ENTRY"] = Field(..., alias="type")
    track_id: int = Field(..., ge=0, alias="trackId")
    region_id: str = Field(..., alias="regionId")
    geometry_version: int = Field(..., ge=1, alias="geometryVersion")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("region_id")
    @classmethod
    def check_region_id(cls, v: str) -> str:
        return validate_uuid_str(v, "regionId")


class IdentityCandidateObservation(StrictWireModel):
    type: Literal["IDENTITY_CANDIDATE"] = Field(..., alias="type")
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


class EvidenceItem(StrictWireModel):
    kind: Literal["FRAME", "CROP", "SNAPSHOT"]
    uri: str = Field(..., min_length=1)
    track_id: int | None = Field(default=None, ge=0, alias="trackId")
    bounding_box: BoundingBox | None = Field(default=None, alias="boundingBox")


class TechnicalObservationEvent(StrictWireModel):
    event_id: str = Field(..., alias="eventId")
    schema_version: Literal["1.0.0"] = Field(..., alias="schemaVersion")
    camera_external_id: str = Field(..., min_length=1, max_length=128, alias="cameraExternalId")
    stream_session_id: str = Field(..., alias="streamSessionId")
    captured_at: str = Field(..., alias="capturedAt")
    frame_dimensions: FrameDimensions = Field(..., alias="frameDimensions")
    observations: list[Observation] = Field(..., min_length=1)
    evidence: list[EvidenceItem] = Field(..., alias="evidence")

    @field_validator("event_id")
    @classmethod
    def check_event_id(cls, v: str) -> str:
        return validate_uuid_str(v, "eventId")

    @field_validator("stream_session_id")
    @classmethod
    def check_stream_session_id(cls, v: str) -> str:
        return validate_uuid_str(v, "streamSessionId")

    @field_validator("captured_at")
    @classmethod
    def check_captured_at(cls, v: str) -> str:
        return validate_rfc3339_captured_at(v)

    def to_wire_dict(self) -> dict[str, Any]:
        """Serializes event to canonical wire dictionary with camelCase keys and no nulls."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def compute_payload_hash(self) -> str:
        """Computes the RFC 8785 canonical SHA-256 hash of this event's wire representation."""
        from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash

        return compute_canonical_payload_hash(self.to_wire_dict())

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        camera_external_id: str,
        stream_session_id: str,
        captured_at: str,
        frame_dimensions: FrameDimensions | dict[str, Any],
        observations: list[Any],
        evidence: list[Any] | None = None,
        schema_version: Literal["1.0.0"] = "1.0.0",
    ) -> "TechnicalObservationEvent":
        """Convenience constructor for internal Python usage, keeping wire validation strict."""
        fd = (
            frame_dimensions
            if isinstance(frame_dimensions, FrameDimensions)
            else FrameDimensions.model_validate(frame_dimensions)
        )
        data = {
            "eventId": event_id,
            "schemaVersion": schema_version,
            "cameraExternalId": camera_external_id,
            "streamSessionId": stream_session_id,
            "capturedAt": captured_at,
            "frameDimensions": fd.to_wire_dict(),
            "observations": [
                obs.to_wire_dict() if hasattr(obs, "to_wire_dict") else obs for obs in observations
            ],
            "evidence": [
                ev.to_wire_dict() if hasattr(ev, "to_wire_dict") else ev for ev in (evidence or [])
            ],
        }
        return cls.model_validate(data)
