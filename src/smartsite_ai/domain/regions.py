import struct
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_CAMERA_REGION_PAYLOAD_BYTES = 262_144


def _validate_json_integer(value: Any, info: ValidationInfo) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer in this wire contract")
    if isinstance(value, int):
        return value
    if info.mode == "json" and isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("value must be an integer JSON number")


JsonInteger = Annotated[int, BeforeValidator(_validate_json_integer)]
NormalizedNumber = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
NormalizedCoordinate = tuple[NormalizedNumber, NormalizedNumber]
ExactCoordinate = tuple[int, int]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        populate_by_name=False,
    )


def _exact_float(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & 0xFFFFFFFFFFFFF
    if exponent == 0:
        return fraction
    return ((1 << 52) | fraction) << (exponent - 1)


def _cross(a: ExactCoordinate, b: ExactCoordinate, c: ExactCoordinate) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: ExactCoordinate, b: ExactCoordinate, point: ExactCoordinate) -> bool:
    return min(a[0], b[0]) <= point[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= point[1] <= max(
        a[1], b[1]
    )


def _segments_intersect(
    a: ExactCoordinate,
    b: ExactCoordinate,
    c: ExactCoordinate,
    d: ExactCoordinate,
) -> bool:
    abc = _cross(a, b, c)
    abd = _cross(a, b, d)
    cda = _cross(c, d, a)
    cdb = _cross(c, d, b)

    if abc == 0 and _on_segment(a, b, c):
        return True
    if abd == 0 and _on_segment(a, b, d):
        return True
    if cda == 0 and _on_segment(c, d, a):
        return True
    if cdb == 0 and _on_segment(c, d, b):
        return True

    return ((abc > 0 and abd < 0) or (abc < 0 and abd > 0)) and (
        (cda > 0 and cdb < 0) or (cda < 0 and cdb > 0)
    )


def _has_zero_area(coordinates: tuple[ExactCoordinate, ...]) -> bool:
    twice_area = 0
    for index, current in enumerate(coordinates):
        following = coordinates[(index + 1) % len(coordinates)]
        twice_area += current[0] * following[1] - following[0] * current[1]
    return twice_area == 0


class RegionPolygon(_StrictFrozenModel):
    coordinates: tuple[NormalizedCoordinate, ...] = Field(min_length=3, max_length=64)

    @field_validator("coordinates")
    @classmethod
    def validate_polygon(
        cls, coordinates: tuple[NormalizedCoordinate, ...]
    ) -> tuple[NormalizedCoordinate, ...]:
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("Polygon vertices must be distinct, including the closing vertex")

        exact = tuple((_exact_float(x), _exact_float(y)) for x, y in coordinates)
        if _has_zero_area(exact):
            raise ValueError("Polygon must have nonzero area")

        for first in range(len(exact)):
            first_next = (first + 1) % len(exact)
            for second in range(first + 1, len(exact)):
                second_next = (second + 1) % len(exact)
                if first_next == second or second_next == first:
                    continue
                if _segments_intersect(
                    exact[first], exact[first_next], exact[second], exact[second_next]
                ):
                    raise ValueError("Non-adjacent polygon edges must not intersect")

        return coordinates


class CameraObservationRegionConfiguration(_StrictFrozenModel):
    region_id: str = Field(alias="regionId")
    geometry_version: JsonInteger = Field(alias="geometryVersion", ge=1, le=MAX_SAFE_INTEGER)
    coordinate_space: Literal["NORMALIZED_0_1"] = Field(alias="coordinateSpace")
    polygon: RegionPolygon

    @field_validator("region_id")
    @classmethod
    def validate_region_id(cls, value: str) -> str:
        if len(value) != 36:
            raise ValueError("regionId must be a canonical UUID string")
        groups = (8, 4, 4, 4, 12)
        parts = value.split("-")
        if len(parts) != len(groups) or any(
            len(part) != expected
            or any(character not in "0123456789abcdefABCDEF" for character in part)
            for part, expected in zip(parts, groups, strict=True)
        ):
            raise ValueError("regionId must be a canonical UUID string")
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError("regionId must be a valid UUID string") from exc
        return value


class CameraRegionConfiguration(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    regions: tuple[CameraObservationRegionConfiguration, ...] = Field(max_length=64)

    @field_validator("camera_external_id")
    @classmethod
    def validate_camera_external_id(cls, value: str) -> str:
        if "\x00" in value or any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("cameraExternalId contains a forbidden character")
        return value

    @model_validator(mode="after")
    def validate_unique_region_ids(self) -> "CameraRegionConfiguration":
        region_ids = [region.region_id.lower() for region in self.regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Region IDs must be unique")
        return self

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> "CameraRegionConfiguration":
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) > MAX_CAMERA_REGION_PAYLOAD_BYTES:
            raise ValueError(
                f"Camera region payload exceeds {MAX_CAMERA_REGION_PAYLOAD_BYTES} UTF-8 bytes"
            )
        return cls.model_validate_json(payload, strict=True)


__all__ = [
    "CameraObservationRegionConfiguration",
    "CameraRegionConfiguration",
    "MAX_CAMERA_REGION_PAYLOAD_BYTES",
    "NormalizedCoordinate",
    "RegionPolygon",
]
