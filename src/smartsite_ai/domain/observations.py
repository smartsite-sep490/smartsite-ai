import re
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


DATE_RE = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$")
TIME_RE = re.compile(
    r"^([0-9]{2}):([0-9]{2}):([0-9]{2}(?:\.[0-9]+)?)([zZ]|([+-])([0-9]{2})(?::?([0-9]{2}))?)?$"
)
DATE_TIME_SEP_RE = re.compile(r"[tT ]")


def validate_uuid_str(v: Any, field_name: str) -> str:
    if not isinstance(v, str) or not UUID_RE.match(v):
        raise ValueError(f"{field_name} must be a valid UUID string")
    try:
        UUID(v)
    except Exception as e:
        raise ValueError(f"Invalid UUID in {field_name}: {e}") from e
    return v


def validate_rfc3339_captured_at(v: Any) -> str:
    """Validates date-time string matching Ajv date-time full format with strict time zone."""
    if not isinstance(v, str):
        raise ValueError("capturedAt must be a string")

    parts = DATE_TIME_SEP_RE.split(v)
    if len(parts) != 2:
        raise ValueError(f"capturedAt '{v}' must contain exactly one date-time separator")

    date_part, time_part = parts

    date_match = DATE_RE.match(date_part)
    if not date_match:
        raise ValueError(f"Invalid date format in capturedAt '{v}'")

    year = int(date_match.group(1))
    month = int(date_match.group(2))
    day = int(date_match.group(3))

    if month < 1 or month > 12:
        raise ValueError(f"Invalid month {month} in capturedAt '{v}'")

    max_days = 29 if (month == 2 and is_leap_year(year)) else DAYS_IN_MONTH[month]
    if day < 1 or day > max_days:
        raise ValueError(f"Invalid day {day} for month {month} in capturedAt '{v}'")

    time_match = TIME_RE.match(time_part)
    if not time_match:
        raise ValueError(f"Invalid time format in capturedAt '{v}'")

    hr = int(time_match.group(1))
    min_ = int(time_match.group(2))
    sec = float(time_match.group(3))
    tz = time_match.group(4)

    if not tz:
        raise ValueError(f"Missing required timezone offset in capturedAt '{v}'")

    tz_sign = -1 if time_match.group(5) == "-" else 1
    tz_h = int(time_match.group(6) or 0)
    tz_m = int(time_match.group(7) or 0)

    if tz_h > 23 or tz_m > 59:
        raise ValueError(f"Timezone offset out of range in capturedAt '{v}': {tz_h:02d}:{tz_m:02d}")

    if hr <= 23 and min_ <= 59 and sec < 60.0:
        return v

    # Leap second handling matching Ajv
    utc_min = min_ - tz_m * tz_sign
    utc_hr = hr - tz_h * tz_sign - (1 if utc_min < 0 else 0)
    if (utc_hr == 23 or utc_hr == -1) and (utc_min == 59 or utc_min == -1) and sec < 61.0:
        return v

    raise ValueError(f"Invalid time or leap-second components in capturedAt '{v}'")


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
