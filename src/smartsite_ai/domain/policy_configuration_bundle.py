from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import (
    MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES,
    MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS,
    PPE_ITEMS,
    CameraConfigurationBundle,
    CameraConfigurationBundleRegion,
    PpeItem,
    _reject_duplicate_json_keys,
)
from smartsite_ai.domain.regions import (
    MAX_SAFE_INTEGER,
    CameraObservationRegionConfiguration,
    CameraRegionConfiguration,
    JsonInteger,
    _StrictFrozenModel,
)

_JS_WHITESPACE = "".join(
    chr(code)
    for code in (
        9,
        10,
        11,
        12,
        13,
        32,
        160,
        5760,
        *range(8192, 8203),
        8232,
        8233,
        8239,
        8287,
        12288,
        65279,
    )
)


class ExplicitPpePolicy(_StrictFrozenModel):
    mode: Literal["EXPLICIT"]
    required_ppe: tuple[PpeItem, ...] = Field(
        alias="requiredPpe", min_length=1, max_length=len(PPE_ITEMS)
    )

    @field_validator("required_ppe")
    @classmethod
    def unique_items(cls, value: tuple[PpeItem, ...]) -> tuple[PpeItem, ...]:
        if len(value) != len(set(value)):
            raise ValueError("requiredPpe items must be unique")
        return value


class DisabledPpePolicy(_StrictFrozenModel):
    mode: Literal["DISABLED"]
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def valid_reason(cls, value: str) -> str:
        if not value.strip(_JS_WHITESPACE):
            raise ValueError("DISABLED requires a nonblank reason")
        if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
            raise ValueError("Policy reason contains control or surrogate characters")
        if len(value.encode("utf-16-le")) // 2 > 500:
            raise ValueError("Policy reason exceeds 500 UTF-16 code units")
        # Transport retains original content so its digest remains unchanged.
        return value


class InheritPpePolicy(_StrictFrozenModel):
    mode: Literal["INHERIT"]


CameraPpePolicy = Annotated[ExplicitPpePolicy | DisabledPpePolicy, Field(discriminator="mode")]
RegionPpePolicy = Annotated[
    ExplicitPpePolicy | DisabledPpePolicy | InheritPpePolicy,
    Field(discriminator="mode"),
]


@dataclass(frozen=True)
class PpePolicyContribution:
    region_id: str | None
    source: Literal["CAMERA_DEFAULT", "REGION"]
    mode: Literal["EXPLICIT", "DISABLED", "INHERIT"]
    required_ppe: tuple[PpeItem, ...]


@dataclass(frozen=True)
class ResolvedLocationPpePolicy:
    scope: Literal["CAMERA", "REGIONS"]
    required_ppe: tuple[PpeItem, ...]
    contributions: tuple[PpePolicyContribution, ...]


class CameraPolicyConfigurationBundleRegion(CameraObservationRegionConfiguration):
    zone_id: str = Field(alias="zoneId")
    policy_revision: JsonInteger = Field(alias="policyRevision", ge=1, le=MAX_SAFE_INTEGER)
    ppe_policy: RegionPpePolicy = Field(alias="ppePolicy")

    @field_validator("region_id")
    @classmethod
    def lowercase_region_id(cls, value: str) -> str:
        return CameraConfigurationBundleRegion.validate_lowercase_region_id(value)

    @field_validator("zone_id")
    @classmethod
    def lowercase_zone_id(cls, value: str) -> str:
        return CameraConfigurationBundleRegion.validate_zone_id(value)


class CameraPolicyConfigurationBundle(_StrictFrozenModel):
    """V2 transport only. Parsing does not apply policy or authorize a change."""

    schema_version: Literal["2.0.0"] = Field(alias="schemaVersion")
    camera_id: str = Field(alias="cameraId")
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    configuration_version: JsonInteger = Field(
        alias="configurationVersion", ge=1, le=MAX_SAFE_INTEGER
    )
    view_revision: JsonInteger = Field(alias="viewRevision", ge=1, le=MAX_SAFE_INTEGER)
    issued_at: str = Field(alias="issuedAt")
    bundle_sha256: str = Field(alias="bundleSha256")
    camera_ppe_policy: CameraPpePolicy = Field(alias="cameraPpePolicy")
    regions: tuple[CameraPolicyConfigurationBundleRegion, ...] = Field(
        max_length=MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS
    )

    @field_validator("camera_id")
    @classmethod
    def canonical_camera_id(cls, value: str) -> str:
        return CameraConfigurationBundle.validate_camera_id(value)

    @field_validator("camera_external_id")
    @classmethod
    def valid_external_id(cls, value: str) -> str:
        return CameraConfigurationBundle.validate_camera_external_id(value)

    @field_validator("issued_at")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        return CameraConfigurationBundle.validate_issued_at(value)

    @field_validator("bundle_sha256")
    @classmethod
    def canonical_digest(cls, value: str) -> str:
        return CameraConfigurationBundle.validate_bundle_sha256(value)

    @model_validator(mode="after")
    def valid_bundle(self) -> Self:
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("Region IDs must be unique")
        content = self.model_dump(mode="json", by_alias=True, exclude={"bundle_sha256"})
        if compute_canonical_payload_hash(content) != self.bundle_sha256:
            raise ValueError("bundleSha256 does not match canonical bundle content")
        return self

    def resolve_ppe_policy(
        self, containing_region_ids: tuple[str, ...]
    ) -> ResolvedLocationPpePolicy:
        """Resolve supplied containing Regions; this does not prove geometry or wearing."""
        if (
            not isinstance(containing_region_ids, tuple)
            or len(containing_region_ids) > MAX_CAMERA_CONFIGURATION_BUNDLE_REGIONS
            or not all(isinstance(region_id, str) for region_id in containing_region_ids)
            or len(set(containing_region_ids)) != len(containing_region_ids)
        ):
            raise ValueError("Invalid containing Region selection")
        policies = {region.region_id: region.ppe_policy for region in self.regions}
        if any(region_id not in policies for region_id in containing_region_ids):
            raise ValueError("Unknown Region in the configuration snapshot")
        selected: list[tuple[str | None, RegionPpePolicy]] = (
            [(region_id, policies[region_id]) for region_id in containing_region_ids]
            if containing_region_ids
            else [(None, self.camera_ppe_policy)]
        )
        contributions = []
        for region_id, policy in selected:
            if region_id is None or isinstance(policy, InheritPpePolicy):
                effective = self.camera_ppe_policy
                source = "CAMERA_DEFAULT"
            else:
                effective = policy
                source = "REGION"
            items = effective.required_ppe if isinstance(effective, ExplicitPpePolicy) else ()
            contributions.append(PpePolicyContribution(region_id, source, policy.mode, items))
        required = tuple(sorted({item for entry in contributions for item in entry.required_ppe}))
        return ResolvedLocationPpePolicy(
            "REGIONS" if containing_region_ids else "CAMERA", required, tuple(contributions)
        )

    def to_region_configuration(self) -> CameraRegionConfiguration:
        """Geometry view only; consumers must retain this bundle for PPE policy."""
        return CameraRegionConfiguration(
            schemaVersion="1.0.0",
            configurationVersion=self.configuration_version,
            cameraExternalId=self.camera_external_id,
            regions=tuple(
                CameraObservationRegionConfiguration(
                    regionId=region.region_id,
                    geometryVersion=region.geometry_version,
                    coordinateSpace=region.coordinate_space,
                    polygon=region.polygon,
                )
                for region in self.regions
            ),
        )

    @classmethod
    def from_wire_bytes(cls, payload: bytes) -> Self:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) > MAX_CAMERA_CONFIGURATION_BUNDLE_BYTES:
            raise ValueError("Camera policy bundle exceeds the wire byte limit")
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
        bundle = cls.model_validate_json(payload, strict=True)
        content = {key: value for key, value in raw.items() if key != "bundleSha256"}
        if compute_canonical_payload_hash(content) != bundle.bundle_sha256:
            raise ValueError("bundleSha256 does not match unmodified wire content")
        return bundle
