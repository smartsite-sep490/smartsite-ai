from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.regions import (
    MAX_CAMERA_REGION_PAYLOAD_BYTES,
    MAX_SAFE_INTEGER,
    CameraObservationRegionConfiguration,
    CameraRegionConfiguration,
    JsonInteger,
    _StrictFrozenModel,
)

MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES = MAX_CAMERA_REGION_PAYLOAD_BYTES
MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS = 64
PPE_ITEMS: tuple[str, ...] = (
    "HARD_HAT",
    "SAFETY_VEST",
    "GLOVES",
    "BOOTS",
    "GOGGLES",
)
PpeItem = Literal["HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"]

_ISSUED_AT_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _validate_lowercase_uuid(value: str, *, field_name: str) -> str:
    CameraObservationRegionConfiguration.validate_region_id(value)
    if value != value.lower():
        raise ValueError(f"{field_name} must be a lowercase canonical UUID string")
    return value


def _validate_utc_millisecond_timestamp(value: str) -> str:
    if _ISSUED_AT_PATTERN.fullmatch(value) is None:
        raise ValueError("issuedAt must be an ISO-8601 UTC millisecond timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("issuedAt must be a valid calendar timestamp") from exc
    formatted = f"{parsed.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z"
    if formatted != value:
        raise ValueError("issuedAt must be a canonical UTC millisecond timestamp")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


class CameraConfigurationBundleRegion(CameraObservationRegionConfiguration):
    zone_id: str = Field(alias="zoneId")
    policy_revision: JsonInteger = Field(alias="policyRevision", ge=1, le=MAX_SAFE_INTEGER)
    required_ppe: tuple[PpeItem, ...] = Field(alias="requiredPpe", max_length=len(PPE_ITEMS))

    @field_validator("region_id")
    @classmethod
    def validate_lowercase_region_id(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="regionId")

    @field_validator("zone_id")
    @classmethod
    def validate_zone_id(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="zoneId")

    @field_validator("required_ppe")
    @classmethod
    def validate_unique_required_ppe(cls, value: tuple[PpeItem, ...]) -> tuple[PpeItem, ...]:
        if len(set(value)) != len(value):
            raise ValueError("requiredPpe items must be unique")
        return value


class CameraConfigurationBundle(_StrictFrozenModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    camera_id: str = Field(alias="cameraId")
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    view_revision: JsonInteger = Field(alias="viewRevision", ge=1, le=MAX_SAFE_INTEGER)
    issued_at: str = Field(alias="issuedAt")
    bundle_sha256: str = Field(alias="bundleSha256")
    regions: tuple[CameraConfigurationBundleRegion, ...] = Field(
        max_length=MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS
    )

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="cameraId")

    @field_validator("camera_external_id")
    @classmethod
    def validate_camera_external_id(cls, value: str) -> str:
        return CameraRegionConfiguration.validate_camera_external_id(value)

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at(cls, value: str) -> str:
        return _validate_utc_millisecond_timestamp(value)

    @field_validator("bundle_sha256")
    @classmethod
    def validate_bundle_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("bundleSha256 must be a 64-character lowercase hex digest")
        return value

    @model_validator(mode="after")
    def validate_bundle_invariants(self) -> CameraConfigurationBundle:
        region_ids = [region.region_id for region in self.regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Region IDs must be unique")
        content = self.model_dump(mode="json", by_alias=True, exclude={"bundle_sha256"})
        if compute_canonical_payload_hash(content) != self.bundle_sha256:
            raise ValueError("bundleSha256 does not match canonical bundle content")
        return self

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> CameraConfigurationBundle:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) > MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES:
            raise ValueError(
                "Camera configuration bundle payload exceeds "
                f"{MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES} UTF-8 bytes"
            )
        bundle = cls.model_validate_json(payload, strict=True)
        raw_data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(raw_data, dict):
            raise ValueError("Camera configuration bundle payload must be a JSON object")
        raw_content = {key: value for key, value in raw_data.items() if key != "bundleSha256"}
        if compute_canonical_payload_hash(raw_content) != bundle.bundle_sha256:
            raise ValueError("bundleSha256 does not match unmodified wire content")
        return bundle


__all__ = [
    "CameraConfigurationBundle",
    "CameraConfigurationBundleRegion",
    "MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES",
    "MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS",
    "PPE_ITEMS",
    "PpeItem",
]
