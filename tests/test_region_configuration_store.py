import asyncio
import json
from typing import Any

import pytest

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationConflict,
    RegionConfigurationStore,
)
from smartsite_ai.domain.regions import CameraRegionConfiguration


def _snapshot(
    version: int,
    *,
    camera_external_id: str = "CAM-GATE-01",
    geometry_version: int = 1,
    apex_x: float = 0.5,
) -> CameraRegionConfiguration:
    payload: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "configurationVersion": version,
        "cameraExternalId": camera_external_id,
        "regions": [
            {
                "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
                "geometryVersion": geometry_version,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0, 0], [1, 0], [apex_x, 1]]},
            }
        ],
    }
    return CameraRegionConfiguration.from_wire_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )


def test_first_apply_is_visible_through_synchronized_get():
    async def scenario():
        store = RegionConfigurationStore()
        snapshot = _snapshot(1)

        assert await store.apply(snapshot) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.get("CAM-GATE-01") is snapshot

    asyncio.run(scenario())


def test_greater_version_replaces_and_lower_version_is_stale():
    async def scenario():
        store = RegionConfigurationStore()
        first = _snapshot(1)
        newest = _snapshot(3, geometry_version=3)
        stale = _snapshot(2, geometry_version=2)

        assert await store.apply(first) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.apply(newest) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.apply(stale) is RegionConfigurationApplyOutcome.STALE
        assert await store.get("CAM-GATE-01") is newest

    asyncio.run(scenario())


def test_equal_version_and_same_canonical_hash_is_idempotent():
    async def scenario():
        store = RegionConfigurationStore()
        original = _snapshot(7)
        equivalent = CameraRegionConfiguration.from_wire_bytes(
            json.dumps(
                original.model_dump(mode="json", by_alias=True),
                indent=2,
            ).encode("utf-8")
        )

        assert await store.apply(original) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.apply(equivalent) is RegionConfigurationApplyOutcome.IDEMPOTENT
        assert await store.get("CAM-GATE-01") is original

    asyncio.run(scenario())


def test_equal_version_and_different_canonical_hash_is_a_conflict():
    async def scenario():
        store = RegionConfigurationStore()
        original = _snapshot(7)
        conflicting = _snapshot(7, geometry_version=2)
        await store.apply(original)

        with pytest.raises(RegionConfigurationConflict, match="CAM-GATE-01.*7"):
            await store.apply(conflicting)
        assert await store.get("CAM-GATE-01") is original

    asyncio.run(scenario())


def test_versions_are_independent_per_camera_and_missing_camera_returns_none():
    async def scenario():
        store = RegionConfigurationStore()
        gate = _snapshot(9, camera_external_id="CAM-GATE")
        yard = _snapshot(2, camera_external_id="CAM-YARD")

        assert await store.get("CAM-MISSING") is None
        assert await store.apply(gate) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.apply(yard) is RegionConfigurationApplyOutcome.APPLIED
        assert await store.get("CAM-GATE") is gate
        assert await store.get("CAM-YARD") is yard

    asyncio.run(scenario())


def test_concurrent_out_of_order_applies_leave_the_greatest_snapshot_active():
    async def scenario():
        store = RegionConfigurationStore()
        snapshots = [_snapshot(version, geometry_version=version) for version in range(1, 17)]
        release = asyncio.Event()

        async def apply_when_released(snapshot: CameraRegionConfiguration):
            await release.wait()
            return await store.apply(snapshot)

        tasks = [asyncio.create_task(apply_when_released(snapshot)) for snapshot in snapshots]
        release.set()
        await asyncio.gather(*reversed(tasks))

        active = await store.get("CAM-GATE-01")
        assert active is not None
        assert active.configuration_version == 16
        assert active.regions[0].geometry_version == 16

    asyncio.run(scenario())


def test_older_apply_finishing_after_newer_apply_cannot_roll_back_snapshot():
    async def scenario():
        store = RegionConfigurationStore()
        newer_applied = asyncio.Event()
        older = _snapshot(1)
        newer = _snapshot(2, geometry_version=2)

        async def apply_newer():
            outcome = await store.apply(newer)
            newer_applied.set()
            return outcome

        async def apply_older_afterward():
            await newer_applied.wait()
            return await store.apply(older)

        newer_outcome, older_outcome = await asyncio.gather(apply_newer(), apply_older_afterward())

        assert newer_outcome is RegionConfigurationApplyOutcome.APPLIED
        assert older_outcome is RegionConfigurationApplyOutcome.STALE
        assert await store.get("CAM-GATE-01") is newer

    asyncio.run(scenario())
