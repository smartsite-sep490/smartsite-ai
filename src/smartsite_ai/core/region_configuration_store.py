import asyncio
from enum import StrEnum

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.regions import CameraRegionConfiguration


class RegionConfigurationApplyOutcome(StrEnum):
    APPLIED = "APPLIED"
    IDEMPOTENT = "IDEMPOTENT"
    STALE = "STALE"


class RegionConfigurationConflict(ValueError):
    """Raised when one camera/version pair is associated with different content."""


class RegionConfigurationStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snapshots: dict[str, tuple[CameraRegionConfiguration, str]] = {}

    async def apply(self, snapshot: CameraRegionConfiguration) -> RegionConfigurationApplyOutcome:
        payload_hash = compute_canonical_payload_hash(
            snapshot.model_dump(mode="json", by_alias=True)
        )

        async with self._lock:
            current = self._snapshots.get(snapshot.camera_external_id)
            if current is None:
                self._snapshots[snapshot.camera_external_id] = (snapshot, payload_hash)
                return RegionConfigurationApplyOutcome.APPLIED

            current_snapshot, current_hash = current
            if snapshot.configuration_version > current_snapshot.configuration_version:
                self._snapshots[snapshot.camera_external_id] = (snapshot, payload_hash)
                return RegionConfigurationApplyOutcome.APPLIED
            if snapshot.configuration_version < current_snapshot.configuration_version:
                return RegionConfigurationApplyOutcome.STALE
            if payload_hash == current_hash:
                return RegionConfigurationApplyOutcome.IDEMPOTENT

            raise RegionConfigurationConflict(
                "Camera region configuration conflict for "
                f"{snapshot.camera_external_id} at version {snapshot.configuration_version}"
            )

    async def get(self, camera_external_id: str) -> CameraRegionConfiguration | None:
        async with self._lock:
            current = self._snapshots.get(camera_external_id)
            return current[0] if current is not None else None


__all__ = [
    "RegionConfigurationApplyOutcome",
    "RegionConfigurationConflict",
    "RegionConfigurationStore",
]
