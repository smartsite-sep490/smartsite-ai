import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest
from test_configuration_bundle_store import bundle
from test_policy_configuration_bundle import content, packet

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationConflict,
    RegionConfigurationStore,
)
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.integrations.backend_configuration_client import ConfigurationBundleResult
from smartsite_ai.runtime.assembly import preflight_manifest


@pytest.mark.parametrize(
    ("profile", "item", "rejected"),
    [(None, "GLOVES", True), (None, "HARD_HAT", False), ("unified-ppe-15", "GLOVES", False)],
)
def test_runtime_preflight_matches_requested_ppe_to_selected_profile(profile, item, rejected):
    async def scenario():
        current = bundle(ppe=item)

        class Client:
            closed = False

            async def fetch_bundle(self, *_args, **_kwargs):
                return ConfigurationBundleResult(bundle=current, etag='"one"')

            async def aclose(self):
                self.closed = True

        client = Client()
        camera = SimpleNamespace(
            stream_id="synthetic-stream",
            camera_external_id=current.camera_external_id,
            entry=SimpleNamespace(
                camera_id=UUID(current.camera_id),
                ppe_region_id=UUID(current.regions[0].region_id),
                generation_credential_file=None,
                tracked_v17=None,
                partitioned_frame_delivery=True,
            ),
        )
        bound = SimpleNamespace(
            cameras=(camera,),
            manifest=SimpleNamespace(
                experimental_model_profile=profile,
                poll_interval_seconds=2,
                stale_after_seconds=10,
            ),
        )
        failure, plans = await preflight_manifest(
            bound, configuration_client_factory=lambda: client
        )
        assert (failure is not None) is rejected
        assert len(plans) == (0 if rejected else 1)
        assert client.closed is rejected
        if not rejected:
            await client.aclose()

    asyncio.run(scenario())


def test_unsupported_policy_update_preserves_installed_bundle_and_geometry():
    async def scenario():
        store = RegionConfigurationStore(supported_ppe_items=frozenset({"HARD_HAT", "SAFETY_VEST"}))
        first = bundle()
        await store.apply_bundle(first)
        projection = await store.get(first.camera_external_id)
        with pytest.raises(RegionConfigurationConflict, match="unsupported PPE"):
            await store.apply_bundle(bundle(2, ppe="GLOVES"))
        assert await store.get_bundle(first.camera_external_id) is first
        assert await store.get(first.camera_external_id) is projection

    asyncio.run(scenario())


def test_v2_checks_camera_and_region_explicit_requirements_but_allows_disabled_policies():
    async def scenario():
        store = RegionConfigurationStore(
            allow_policy_v2=True, supported_ppe_items=frozenset({"HARD_HAT", "SAFETY_VEST"})
        )
        value = content()
        with pytest.raises(RegionConfigurationConflict, match="unsupported PPE"):
            await store.apply_bundle(CameraPolicyConfigurationBundle.from_wire_bytes(packet(value)))
        assert await store.get_bundle("camera-01") is None
        value["regions"][1]["ppePolicy"] = {"mode": "DISABLED", "reason": "Synthetic exception"}
        core = CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))
        assert await store.apply_bundle(core) is RegionConfigurationApplyOutcome.APPLIED
        value["configurationVersion"] += 1
        value["cameraPpePolicy"] = {"mode": "EXPLICIT", "requiredPpe": ["BOOTS"]}
        with pytest.raises(RegionConfigurationConflict, match="unsupported PPE"):
            await store.apply_bundle(CameraPolicyConfigurationBundle.from_wire_bytes(packet(value)))
        assert await store.get_bundle("camera-01") is core

    asyncio.run(scenario())


def test_zone_only_policy_needs_no_supported_ppe():
    async def scenario():
        store = RegionConfigurationStore(allow_policy_v2=True, supported_ppe_items=frozenset())
        value = content()
        value["cameraPpePolicy"] = {"mode": "DISABLED", "reason": "Zone-only camera"}
        for region in value["regions"]:
            region["ppePolicy"] = {"mode": "INHERIT"}
        assert (
            await store.apply_bundle(CameraPolicyConfigurationBundle.from_wire_bytes(packet(value)))
            is RegionConfigurationApplyOutcome.APPLIED
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["HARD_HAT", frozenset({"MASK"}), frozenset({True})])
def test_invalid_capability_bounds_fail_closed(invalid):
    with pytest.raises(ValueError):
        RegionConfigurationStore(supported_ppe_items=invalid)
