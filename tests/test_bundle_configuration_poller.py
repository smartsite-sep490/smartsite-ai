import json
from uuid import UUID

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.integrations.backend_configuration_client import (
    ConfigurationBundleResult,
    ConfigurationNotModifiedResult,
)
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller


def make_bundle(version, ppe):
    content = {
        "schemaVersion": "1.0.0",
        "cameraId": str(UUID(int=1)),
        "cameraExternalId": "CAM-1",
        "configurationVersion": version,
        "viewRevision": 1,
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


class BundleClient:
    def __init__(self, results):
        self.results = iter(results)
        self.etags = []

    async def fetch(self, *_args, **_kwargs):
        raise AssertionError("Poller must not use the geometry-only endpoint")

    async def fetch_bundle(self, camera_id, *, expected_camera_external_id, etag=None):
        assert camera_id == UUID(int=1)
        assert expected_camera_external_id == "CAM-1"
        self.etags.append(etag)
        return next(self.results)


@pytest.mark.anyio
async def test_runtime_poller_loads_whole_policy_bundle_and_304_preserves_object_identity():
    first = make_bundle(1, "HARD_HAT")
    client = BundleClient(
        [ConfigurationBundleResult(first, '"1"'), ConfigurationNotModifiedResult('"1"')]
    )
    store = RegionConfigurationStore()
    poller = CameraConfigurationPoller(
        client=client, store=store, camera_id=UUID(int=1), camera_external_id="CAM-1"
    )
    await poller.refresh()
    projection = await store.get("CAM-1")
    assert await store.get_bundle("CAM-1") is first
    await poller.refresh()
    assert await store.get_bundle("CAM-1") is first
    assert await store.get("CAM-1") is projection
    assert client.etags == [None, '"1"']


@pytest.mark.anyio
async def test_policy_only_refresh_changes_frame_snapshot_even_when_geometry_is_unchanged():
    first = make_bundle(1, "HARD_HAT")
    second = make_bundle(2, "GOGGLES")
    store = RegionConfigurationStore()
    poller = CameraConfigurationPoller(
        client=BundleClient(
            [ConfigurationBundleResult(first, '"1"'), ConfigurationBundleResult(second, '"2"')]
        ),
        store=store,
        camera_id=UUID(int=1),
        camera_external_id="CAM-1",
    )
    await poller.refresh()
    old_frame = await store.get("CAM-1")
    await poller.refresh()
    assert await store.get("CAM-1") is not old_frame
    assert (await store.get_bundle("CAM-1")).regions[0].required_ppe == ("GOGGLES",)
