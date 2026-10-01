from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

import pytest

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationStore,
)
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationTransientError,
    ConfigurationFetchResult,
    ConfigurationNotModifiedResult,
    ConfigurationSnapshotResult,
)
from smartsite_ai.integrations.configuration_poller import (
    CameraConfigurationPoller,
    ConfigurationRegressionError,
    ConfigurationStaleError,
)

CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")


def _configuration(version: int) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": version,
            "cameraExternalId": "CAM-GATE-01",
            "regions": (),
        }
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class FakeClient:
    def __init__(self, results: Iterable[ConfigurationFetchResult | Exception]) -> None:
        self._results = iter(results)
        self.etags: list[str | None] = []

    async def fetch(
        self,
        camera_id: UUID,
        *,
        expected_camera_external_id: str,
        etag: str | None = None,
    ) -> ConfigurationFetchResult:
        assert camera_id == CAMERA_ID
        assert expected_camera_external_id == "CAM-GATE-01"
        self.etags.append(etag)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.anyio
async def test_poller_applies_snapshot_then_uses_etag_for_not_modified() -> None:
    client = FakeClient(
        [
            ConfigurationSnapshotResult(_configuration(1), '"v1"'),
            ConfigurationNotModifiedResult('"v1"'),
        ]
    )
    store = RegionConfigurationStore()
    clock = FakeClock()
    poller = CameraConfigurationPoller(
        client=client,  # type: ignore[arg-type]
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        poll_interval_seconds=5,
        stale_after_seconds=30,
        monotonic_clock=clock,
    )

    assert await poller.refresh() is RegionConfigurationApplyOutcome.APPLIED
    clock.value += 5
    assert await poller.refresh() is RegionConfigurationApplyOutcome.IDEMPOTENT
    assert client.etags == [None, '"v1"']
    assert poller.etag == '"v1"'
    assert (await store.get("CAM-GATE-01")).configuration_version == 1  # type: ignore[union-attr]
    poller.ensure_fresh()


@pytest.mark.anyio
async def test_transient_error_uses_last_snapshot_only_within_stale_window() -> None:
    transient = BackendConfigurationTransientError(attempts=4, status_code=503)
    client = FakeClient(
        [
            ConfigurationSnapshotResult(_configuration(1), '"v1"'),
            transient,
            transient,
        ]
    )
    clock = FakeClock()
    poller = CameraConfigurationPoller(
        client=client,  # type: ignore[arg-type]
        store=RegionConfigurationStore(),
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        poll_interval_seconds=5,
        stale_after_seconds=10,
        monotonic_clock=clock,
    )

    await poller.refresh()
    clock.value += 10
    assert await poller.refresh() is None
    clock.value += 0.001
    with pytest.raises(ConfigurationStaleError, match="stale window"):
        await poller.refresh()


@pytest.mark.anyio
async def test_initial_transient_failure_is_fail_closed() -> None:
    poller = CameraConfigurationPoller(
        client=FakeClient(  # type: ignore[arg-type]
            [BackendConfigurationTransientError(attempts=4, status_code=503)]
        ),
        store=RegionConfigurationStore(),
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        poll_interval_seconds=5,
        stale_after_seconds=10,
    )

    with pytest.raises(ConfigurationStaleError, match="never been confirmed"):
        await poller.refresh()


@pytest.mark.anyio
async def test_sequential_older_snapshot_is_rejected_as_backend_regression() -> None:
    client = FakeClient(
        [
            ConfigurationSnapshotResult(_configuration(2), '"v2"'),
            ConfigurationSnapshotResult(_configuration(1), '"v1"'),
        ]
    )
    store = RegionConfigurationStore()
    poller = CameraConfigurationPoller(
        client=client,  # type: ignore[arg-type]
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        poll_interval_seconds=5,
        stale_after_seconds=10,
    )

    await poller.refresh()
    with pytest.raises(ConfigurationRegressionError, match="older"):
        await poller.refresh()
    assert (await store.get("CAM-GATE-01")).configuration_version == 2  # type: ignore[union-attr]


def test_poller_validates_timing_contract() -> None:
    client = FakeClient([])
    with pytest.raises(ValueError, match="positive"):
        CameraConfigurationPoller(
            client=client,  # type: ignore[arg-type]
            store=RegionConfigurationStore(),
            camera_id=CAMERA_ID,
            camera_external_id="CAM-GATE-01",
            poll_interval_seconds=0,
        )
    with pytest.raises(ValueError, match="at least"):
        CameraConfigurationPoller(
            client=client,  # type: ignore[arg-type]
            store=RegionConfigurationStore(),
            camera_id=CAMERA_ID,
            camera_external_id="CAM-GATE-01",
            poll_interval_seconds=10,
            stale_after_seconds=5,
        )
