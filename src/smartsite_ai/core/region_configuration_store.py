import asyncio
from enum import StrEnum

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.regions import (
    CameraObservationRegionConfiguration,
    CameraRegionConfiguration,
)


class RegionConfigurationApplyOutcome(StrEnum):
    APPLIED = "APPLIED"
    IDEMPOTENT = "IDEMPOTENT"
    STALE = "STALE"


class RegionConfigurationConflict(ValueError):
    """Raised when one camera/version pair is associated with different content."""


class RegionConfigurationStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snapshots: dict[
            str, tuple[CameraRegionConfiguration, str, CameraConfigurationBundle | None]
        ] = {}

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

    async def apply_bundle(
        self, bundle: CameraConfigurationBundle
    ) -> RegionConfigurationApplyOutcome:
        projection = CameraRegionConfiguration(
            schemaVersion=bundle.schema_version,
            configurationVersion=bundle.configuration_version,
            cameraExternalId=bundle.camera_external_id,
            regions=tuple(
                CameraObservationRegionConfiguration(
                    regionId=region.region_id,
                    geometryVersion=region.geometry_version,
                    coordinateSpace=region.coordinate_space,
                    polygon=region.polygon,
                )
                for region in bundle.regions
            ),
        )
        async with self._lock:
            current = self._snapshots.get(bundle.camera_external_id)
            if current is not None:
                old_projection, old_hash, old_bundle = current
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

    async def get_bundle(self, camera_external_id: str) -> CameraConfigurationBundle | None:
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            return current[2] if current is not None else None

    async def get(self, camera_external_id: str) -> CameraRegionConfiguration | None:
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            return current[0] if current is not None else None


__all__ = [
    "RegionConfigurationApplyOutcome",
    "RegionConfigurationConflict",
    "RegionConfigurationStore",
]
