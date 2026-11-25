import asyncio
from enum import StrEnum

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import PPE_ITEMS, CameraConfigurationBundle, PpeItem
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.domain.regions import (
    CameraRegionConfiguration,
)


class RegionConfigurationApplyOutcome(StrEnum):
    APPLIED = "APPLIED"
    IDEMPOTENT = "IDEMPOTENT"
    STALE = "STALE"


class RegionConfigurationConflict(ValueError):
    """Raised when one camera/version pair is associated with different content."""


CameraBundle = CameraConfigurationBundle | CameraPolicyConfigurationBundle


class RegionConfigurationStore:
    def __init__(
        self,
        *,
        allow_policy_v2: bool = False,
        supported_ppe_items: frozenset[PpeItem] | None = None,
    ) -> None:
        if not isinstance(allow_policy_v2, bool):
            raise ValueError("Policy v2 support must be an explicit boolean")
        if supported_ppe_items is not None and (
            not isinstance(supported_ppe_items, frozenset)
            or not supported_ppe_items.issubset(PPE_ITEMS)
        ):
            raise ValueError("Supported PPE items must be a canonical immutable set")
        self._allow_policy_v2 = allow_policy_v2
        self._supported_ppe_items = supported_ppe_items
        self._lock = asyncio.Lock()
        self._snapshots: dict[str, tuple[CameraRegionConfiguration, str, CameraBundle | None]] = {}

    async def apply(self, snapshot: CameraRegionConfiguration) -> RegionConfigurationApplyOutcome:
        payload_hash = compute_canonical_payload_hash(
            snapshot.model_dump(mode="json", by_alias=True)
        )

        async with self._lock:
            current = self._snapshots.get(snapshot.camera_external_id)
            if current is None:
                self._snapshots[snapshot.camera_external_id] = (snapshot, payload_hash, None)
                return RegionConfigurationApplyOutcome.APPLIED

            current_snapshot, current_hash, current_bundle = current
            if current_bundle is not None:
                raise RegionConfigurationConflict(
                    "Geometry-only snapshot cannot replace a policy bundle"
                )
            if snapshot.configuration_version > current_snapshot.configuration_version:
                self._snapshots[snapshot.camera_external_id] = (snapshot, payload_hash, None)
                return RegionConfigurationApplyOutcome.APPLIED
            if snapshot.configuration_version < current_snapshot.configuration_version:
                return RegionConfigurationApplyOutcome.STALE
            if payload_hash == current_hash:
                return RegionConfigurationApplyOutcome.IDEMPOTENT

            raise RegionConfigurationConflict(
                "Camera region configuration conflict for "
                f"{snapshot.camera_external_id} at version {snapshot.configuration_version}"
            )

    async def apply_bundle(self, bundle: CameraBundle) -> RegionConfigurationApplyOutcome:
        if isinstance(bundle, CameraPolicyConfigurationBundle) and not self._allow_policy_v2:
            raise RegionConfigurationConflict("Policy bundle v2 requires explicit consumer support")
        if self._supported_ppe_items is not None:
            if isinstance(bundle, CameraPolicyConfigurationBundle):
                policies = (
                    bundle.camera_ppe_policy,
                    *(region.ppe_policy for region in bundle.regions),
                )
                required = {
                    item
                    for policy in policies
                    if policy.mode == "EXPLICIT"
                    for item in policy.required_ppe
                }
            else:
                required = {item for region in bundle.regions for item in region.required_ppe}
            unsupported = required - self._supported_ppe_items
            if unsupported:
                raise RegionConfigurationConflict(
                    "Configuration requests unsupported PPE items: "
                    + ", ".join(sorted(unsupported))
                )
        projection = bundle.to_region_configuration()
        async with self._lock:
            current = self._snapshots.get(bundle.camera_external_id)
            if current is not None:
                old_projection, old_hash, old_bundle = current
                if isinstance(old_bundle, CameraPolicyConfigurationBundle) and not isinstance(
                    bundle, CameraPolicyConfigurationBundle
                ):
                    raise RegionConfigurationConflict("Legacy bundle cannot replace v2 policy")
                if old_bundle is not None and old_bundle.camera_id != bundle.camera_id:
                    raise RegionConfigurationConflict("Bundle camera identity changed")
                if bundle.configuration_version < old_projection.configuration_version or (
                    old_bundle is not None and bundle.view_revision < old_bundle.view_revision
                ):
                    return RegionConfigurationApplyOutcome.STALE
                if bundle.configuration_version == old_projection.configuration_version:
                    if old_bundle is not None:
                        if bundle.bundle_sha256 == old_hash:
                            return RegionConfigurationApplyOutcome.IDEMPOTENT
                        raise RegionConfigurationConflict(
                            "Bundle content changed at the same version"
                        )
                    projection_hash = compute_canonical_payload_hash(
                        projection.model_dump(mode="json", by_alias=True)
                    )
                    if projection_hash != old_hash:
                        raise RegionConfigurationConflict(
                            "Bundle geometry conflicts with saved snapshot"
                        )
            self._snapshots[bundle.camera_external_id] = (projection, bundle.bundle_sha256, bundle)
            return RegionConfigurationApplyOutcome.APPLIED

    async def get_bundle(self, camera_external_id: str) -> CameraBundle | None:
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            return current[2] if current is not None else None

    async def get(self, camera_external_id: str) -> CameraRegionConfiguration | None:
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            return current[0] if current is not None else None

    async def get_active_configuration(
        self, camera_external_id: str
    ) -> CameraRegionConfiguration | CameraBundle | None:
        """Read one immutable frame-boundary snapshot, retaining policy when present."""
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            if current is None:
                return None
            return current[2] if current[2] is not None else current[0]


__all__ = [
    "RegionConfigurationApplyOutcome",
    "RegionConfigurationConflict",
    "RegionConfigurationStore",
]
