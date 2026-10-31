from __future__ import annotations

import json
from collections.abc import Iterable
from uuid import UUID

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationStore,
)
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationError,
    BackendConfigurationHttpError,
    BackendConfigurationNotFoundError,
    BackendConfigurationTransientError,
    ConfigurationBundleFetchResult,
    ConfigurationBundleResult,
    ConfigurationNotModifiedResult,
)
from smartsite_ai.integrations.configuration_poller import (
    CameraConfigurationPoller,
    ConfigurationRegressionError,
    ConfigurationStaleError,
)

CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.mark.parametrize("field", ["poll_interval_seconds", "stale_after_seconds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_poller_rejects_nonfinite_timing(field, value):
    with pytest.raises(ValueError):
        CameraConfigurationPoller(
            client=FakeClient([]),
            store=RegionConfigurationStore(),
            camera_id=CAMERA_ID,
            camera_external_id="CAM-GATE-01",
            **{field: value},
        )


@pytest.mark.anyio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_invalid_clock_cannot_confirm_or_keep_configuration_fresh(value):
    clock = FakeClock()
    store = RegionConfigurationStore()
    client = FakeClient([ConfigurationBundleResult(_configuration(1), '"v1"')])
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
    )
    clock.value = value
    with pytest.raises(ConfigurationStaleError):
        await poller.refresh()
    assert await store.get_active_configuration("CAM-GATE-01") is None
    with pytest.raises(ConfigurationStaleError):
        poller.ensure_fresh()


@pytest.mark.anyio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 99.0])
@pytest.mark.parametrize("refresh", [False, True])
async def test_invalid_clock_revokes_previously_confirmed_freshness(value, refresh):
    clock = FakeClock()
    poller = CameraConfigurationPoller(
        client=FakeClient(
            [
                ConfigurationBundleResult(_configuration(1), '"v1"'),
                ConfigurationNotModifiedResult('"v1"'),
            ]
        ),
        store=RegionConfigurationStore(),
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
    )
    await poller.refresh()
    clock.value = value
    with pytest.raises(ConfigurationStaleError):
        if refresh:
            await poller.refresh()
        else:
            poller.ensure_fresh()
    assert poller.etag is None
    clock.value = 100.0
    with pytest.raises(ConfigurationStaleError, match="never been confirmed"):
        poller.ensure_fresh()


@pytest.mark.anyio
@pytest.mark.parametrize("refresh", [False, True])
async def test_clock_rollback_above_last_refresh_revokes_until_a_fresh_fetch(refresh):
    clock = FakeClock()
    client = FakeClient(
        [
            ConfigurationBundleResult(_configuration(1), '"v1"'),
            *([ConfigurationNotModifiedResult('"v1"')] if refresh else []),
            ConfigurationBundleResult(_configuration(1), '"v1"'),
        ]
    )
    poller = CameraConfigurationPoller(
        client=client,
        store=RegionConfigurationStore(),
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
        stale_after_seconds=120,
    )
    await poller.refresh()
    clock.value = 150.0
    poller.ensure_fresh()
    clock.value = 140.0
    with pytest.raises(ConfigurationStaleError, match="moved backwards"):
        if refresh:
            await poller.refresh()
        else:
            poller.ensure_fresh()
    assert poller.etag is None
    clock.value = 151.0
    with pytest.raises(ConfigurationStaleError, match="never been confirmed"):
        poller.ensure_fresh()
    await poller.refresh()
    assert client.etags[-1] is None
    poller.ensure_fresh()


def _configuration(version: int) -> CameraConfigurationBundle:
    content = {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA_ID),
        "cameraExternalId": "CAM-GATE-01",
        "configurationVersion": version,
        "viewRevision": 1,
        "issuedAt": "2026-10-27T00:00:00.000Z",
        "regions": [],
    }
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode(
            "utf-8"
        )
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class FakeClient:
    def __init__(self, results: Iterable[ConfigurationBundleFetchResult | Exception]) -> None:
        self._results = iter(results)
        self.etags: list[str | None] = []

    async def fetch_bundle(
        self,
        camera_id: UUID,
        *,
        expected_camera_external_id: str,
        etag: str | None = None,
    ) -> ConfigurationBundleFetchResult:
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
            ConfigurationBundleResult(_configuration(1), '"v1"'),
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
            ConfigurationBundleResult(_configuration(1), '"v1"'),
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
            ConfigurationBundleResult(_configuration(2), '"v2"'),
            ConfigurationBundleResult(_configuration(1), '"v1"'),
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    [
        BackendConfigurationNotFoundError("not available"),
        BackendConfigurationHttpError(401),
        BackendConfigurationHttpError(403),
        ConfigurationBundleResult(_configuration(1), '"v1"'),
    ],
)
async def test_hard_refresh_failure_invalidates_guard_and_validator_until_new_snapshot(failure):
    client = FakeClient(
        [
            ConfigurationBundleResult(_configuration(2), '"v2"'),
            failure,
            ConfigurationBundleResult(_configuration(3), '"v3"'),
        ]
    )
    poller = CameraConfigurationPoller(
        client=client,
        store=RegionConfigurationStore(),
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=FakeClock(),
    )
    await poller.refresh()
    with pytest.raises((BackendConfigurationError, ConfigurationRegressionError)):
        await poller.refresh()
    with pytest.raises(ConfigurationStaleError):
        poller.ensure_fresh()
    assert poller.etag is None
    await poller.refresh()
    assert client.etags == [None, '"v2"', None]
    poller.ensure_fresh()
