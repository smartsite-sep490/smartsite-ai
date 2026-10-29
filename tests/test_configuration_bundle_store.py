import asyncio
import json
from uuid import UUID

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome as Outcome,
)
from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationConflict,
    RegionConfigurationStore,
)
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle


def bundle(version=1, *, view=1, ppe="HARD_HAT", camera_id=1):
    content = {
        "schemaVersion": "1.0.0",
        "cameraId": str(UUID(int=camera_id)),
        "cameraExternalId": "CAM-1",
        "configurationVersion": version,
        "viewRevision": view,
        "issuedAt": "2026-10-27T00:00:00.000Z",
        "regions": [
            {
                "regionId": str(UUID(int=2)),
                "zoneId": str(UUID(int=3)),
                "geometryVersion": 1,
                "policyRevision": version,
                "requiredPpe": [ppe],
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0, 0], [1, 0], [0, 1]]},
            }
        ],
    }
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps(
            {
                **content,
                "bundleSha256": compute_canonical_payload_hash(content),
            }
        ).encode()
    )


def test_whole_bundle_is_atomic_and_policy_only_change_replaces_stable_projection():
    async def scenario():
        store = RegionConfigurationStore()
        first = bundle()
        assert await store.apply_bundle(first) is Outcome.APPLIED
        projection = await store.get("CAM-1")
        assert await store.get_bundle("CAM-1") is first
        assert await store.get("CAM-1") is projection
        assert await store.apply_bundle(first) is Outcome.IDEMPOTENT
        assert await store.get("CAM-1") is projection
        newer = bundle(2, ppe="GOGGLES")
        assert await store.apply_bundle(newer) is Outcome.APPLIED
        assert await store.get("CAM-1") is not projection
        assert (await store.get_bundle("CAM-1")).regions[0].required_ppe == ("GOGGLES",)

    asyncio.run(scenario())


def test_stale_views_conflicts_and_legacy_geometry_cannot_remove_loaded_policy():
    async def scenario():
        store = RegionConfigurationStore()
        current = bundle(2, view=2)
        await store.apply_bundle(current)
        projection = await store.get("CAM-1")
        assert await store.apply_bundle(bundle(1, view=2)) is Outcome.STALE
        assert await store.apply_bundle(bundle(3, view=1)) is Outcome.STALE
        for conflict in (bundle(2, view=2, ppe="GOGGLES"), bundle(3, view=2, camera_id=4)):
            with pytest.raises(RegionConfigurationConflict):
                await store.apply_bundle(conflict)
        with pytest.raises(RegionConfigurationConflict):
            await store.apply(projection)
        assert await store.get_bundle("CAM-1") is current
        assert await store.get("CAM-1") is projection

    asyncio.run(scenario())


def test_legacy_snapshot_can_upgrade_without_same_version_geometry_conflict():
    async def scenario():
        source = RegionConfigurationStore()
        current = bundle()
        await source.apply_bundle(current)
        projection = await source.get("CAM-1")
        store = RegionConfigurationStore()
        await store.apply(projection)
        assert await store.get_bundle("CAM-1") is None
        assert await store.apply_bundle(current) is Outcome.APPLIED
        assert await store.get_bundle("CAM-1") is current

    asyncio.run(scenario())
