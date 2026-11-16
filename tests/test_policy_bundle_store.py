import asyncio
from typing import Any

import pytest
from test_policy_configuration_bundle import content, packet

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome as Outcome,
)
from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationConflict,
    RegionConfigurationStore,
)
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle


def bundle(version=4, *, view=2, item="HARD_HAT", apex=0.5, **changes: Any):
    value = content()
    value.update(configurationVersion=version, viewRevision=view, **changes)
    value["cameraPpePolicy"] = {"mode": "EXPLICIT", "requiredPpe": [item]}
    value["regions"][0]["polygon"]["coordinates"][-1][0] = apex
    return CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))


def legacy(version=3):
    value = content()
    value.update(schemaVersion="1.0.0", configurationVersion=version)
    value.pop("cameraPpePolicy")
    for region in value["regions"]:
        region.pop("ppePolicy")
        region["requiredPpe"] = []
    return CameraConfigurationBundle.from_wire_bytes(packet(value))


def test_v2_requires_explicit_consumer_support_and_preserves_existing_state():
    async def scenario():
        store = RegionConfigurationStore()
        first = legacy()
        await store.apply_bundle(first)
        with pytest.raises(RegionConfigurationConflict, match="explicit"):
            await store.apply_bundle(bundle())
        assert await store.get_active_configuration("camera-01") is first

    asyncio.run(scenario())


def test_opted_in_store_retains_full_policy_and_installs_geometry_atomically():
    async def scenario():
        store = RegionConfigurationStore(allow_policy_v2=True)
        first = bundle()
        assert await store.apply_bundle(first) is Outcome.APPLIED
        assert await store.get_active_configuration("camera-01") is first
        assert await store.get_bundle("camera-01") is first
        projection = await store.get("camera-01")
        assert projection.configuration_version == first.configuration_version
        assert projection.schema_version == "1.0.0"
        assert await store.apply_bundle(first) is Outcome.IDEMPOTENT
        assert await store.get("camera-01") is projection
        newer = bundle(5, item="BOOTS")
        assert await store.apply_bundle(newer) is Outcome.APPLIED
        assert first.camera_ppe_policy.required_ppe == ("HARD_HAT",)
        assert await store.get_active_configuration("camera-01") is newer
        assert newer.camera_ppe_policy.required_ppe == ("BOOTS",)
        assert await store.get_active_configuration("unconfigured") is None

    asyncio.run(scenario())


def test_conflicts_stale_views_and_downgrades_cannot_drop_v2_policy():
    async def scenario():
        store = RegionConfigurationStore(allow_policy_v2=True)
        first = bundle()
        await store.apply_bundle(first)
        assert await store.apply_bundle(bundle(3)) is Outcome.STALE
        assert await store.apply_bundle(bundle(5, view=1)) is Outcome.STALE
        for changed in (
            bundle(item="BOOTS"),
            bundle(5, cameraId="11111111-2222-4333-8444-555555555556"),
            legacy(5),
        ):
            with pytest.raises(RegionConfigurationConflict):
                await store.apply_bundle(changed)
        with pytest.raises(RegionConfigurationConflict):
            await store.apply(await store.get("camera-01"))
        assert await store.get_active_configuration("camera-01") is first
        other = bundle(
            cameraExternalId="camera-02", cameraId="11111111-2222-4333-8444-555555555557"
        )
        assert await store.apply_bundle(other) is Outcome.APPLIED
        assert await store.get_active_configuration("camera-01") is first
        assert await store.get_active_configuration("camera-02") is other

    asyncio.run(scenario())


def test_frame_snapshot_is_consistent_during_policy_and_geometry_replacement():
    async def scenario():
        store = RegionConfigurationStore(allow_policy_v2=True)
        first = bundle()
        await store.apply_bundle(first)

        async def writer():
            for version in range(5, 15):
                await store.apply_bundle(
                    bundle(
                        version,
                        item="BOOTS" if version % 2 else "HARD_HAT",
                        apex=0.4 if version % 2 else 0.5,
                    )
                )
                await asyncio.sleep(0)

        async def reader():
            for _ in range(40):
                snapshot = await store.get_active_configuration("camera-01")
                odd = snapshot.configuration_version % 2
                assert snapshot.camera_ppe_policy.required_ppe == ("BOOTS" if odd else "HARD_HAT",)
                assert snapshot.regions[0].polygon.coordinates[-1][0] == (0.4 if odd else 0.5)
                await asyncio.sleep(0)

        await asyncio.gather(writer(), reader())
        assert first.configuration_version == 4
        assert first.regions[0].polygon.coordinates[-1][0] == 0.5

    asyncio.run(scenario())
