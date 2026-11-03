import asyncio
import json
from datetime import timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from test_configuration_poller import CAMERA_ID, FakeClock
from test_worker_generation_binding import NOW, FakeSource, credential_payload

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationHttpError,
    BackendConfigurationTransientError,
    ConfigurationBundleResult,
)
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.worker_generation import (
    GenerationBoundSource,
    WorkerGenerationCredential,
)


def _configuration(version):
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
        json.dumps(
            {
                **content,
                "bundleSha256": compute_canonical_payload_hash(content),
            }
        ).encode()
    )


async def setup_ack(*, generation_clock=lambda: NOW, poll_interval=5):
    client = AsyncMock()
    clock = FakeClock()
    store = RegionConfigurationStore()
    await store.apply_bundle(_configuration(1))
    source = GenerationBoundSource(
        FakeSource(),
        WorkerGenerationCredential.model_validate(credential_payload()),
        clock=generation_clock,
    )
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
        poll_interval_seconds=poll_interval,
        require_bundle=True,
    )
    poller.enable_generation_ack(source)
    client.fetch_bundle.return_value = ConfigurationBundleResult(_configuration(1), '"v1"')
    await poller.refresh()
    return client, clock, store, source, poller


@pytest.mark.anyio
async def test_generation_expiry_invalidates_fresh_metadata_guard():
    now = [NOW]
    _, _, _, _, poller = await setup_ack(generation_clock=lambda: now[0])
    poller.ensure_fresh()
    now[0] += timedelta(hours=1)
    with pytest.raises(RuntimeError, match="generation expired"):
        poller.ensure_fresh()


@pytest.mark.anyio
async def test_transient_fetch_cannot_keep_expired_generation_running():
    now = [NOW]
    client, _, _, source, poller = await setup_ack(generation_clock=lambda: now[0])
    await source.connect()
    await source.read_frame()
    await poller.acknowledge_loaded()
    client.fetch_bundle.side_effect = BackendConfigurationTransientError(attempts=1)
    now[0] += timedelta(hours=1)
    try:
        with pytest.raises(RuntimeError, match="generation expired"):
            await poller.refresh()
        assert client.acknowledge_configuration.await_count == 1
    finally:
        await source.close()


@pytest.mark.anyio
async def test_expiry_during_backend_outage_cancels_owned_worker_and_closes_source():
    from smartsite_ai.tools.run_camera_worker import _coordinate

    now = [NOW]
    client, _, _, source, poller = await setup_ack(
        generation_clock=lambda: now[0], poll_interval=0.01
    )
    closed = asyncio.Event()

    class BlockedWorker:
        async def run(self):
            await source.connect()
            await source.read_frame()
            try:
                await asyncio.Event().wait()
            finally:
                await source.close()
                closed.set()

    async def outage(*args, **kwargs):
        now[0] = NOW + timedelta(hours=1)
        raise BackendConfigurationTransientError(attempts=1)

    client.fetch_bundle.side_effect = outage
    with pytest.raises(RuntimeError, match="generation expired"):
        await asyncio.wait_for(_coordinate(BlockedWorker(), poller), timeout=0.5)
    assert closed.is_set() and not source.is_connected


@pytest.mark.anyio
async def test_ack_requires_bound_frame_and_stops_after_close():
    client, _, _, source, poller = await setup_ack()
    await poller.acknowledge_loaded()
    await source.connect()
    await poller.acknowledge_loaded()
    client.acknowledge_configuration.assert_not_awaited()
    await source.read_frame()
    await poller.acknowledge_loaded()
    ack = client.acknowledge_configuration.call_args.kwargs["acknowledgement"]
    assert ack.configuration_version == 1
    assert ack.bundle_sha256 == _configuration(1).bundle_sha256
    assert ack.stream_session_id == source.binding.stream_session_id
    await source.close()
    await poller.acknowledge_loaded()
    assert client.acknowledge_configuration.await_count == 1


@pytest.mark.anyio
async def test_uncertain_retry_keeps_command_then_new_heartbeat_and_bundle():
    client, clock, store, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()
    client.acknowledge_configuration.side_effect = [
        BackendConfigurationTransientError(attempts=1),
        None,
        None,
        None,
    ]
    await poller.acknowledge_loaded()
    await poller.acknowledge_loaded()
    calls = client.acknowledge_configuration.call_args_list
    first = calls[0].kwargs["acknowledgement"]
    assert calls[1].kwargs["acknowledgement"] == first
    await poller.acknowledge_loaded()
    assert client.acknowledge_configuration.await_count == 2
    clock.value += 30
    await poller.acknowledge_loaded()
    assert (
        client.acknowledge_configuration.call_args.kwargs["acknowledgement"].command_id
        != first.command_id
    )
    await store.apply_bundle(_configuration(2))
    await poller.acknowledge_loaded()
    assert (
        client.acknowledge_configuration.call_args.kwargs["acknowledgement"].configuration_version
        == 2
    )


@pytest.mark.anyio
async def test_terminal_ack_error_propagates_to_owned_poller():
    client, _, _, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()
    client.acknowledge_configuration.side_effect = BackendConfigurationHttpError(403)
    with pytest.raises(BackendConfigurationHttpError):
        await poller.acknowledge_loaded()


@pytest.mark.anyio
async def test_owned_poller_sends_bound_snapshot_through_real_http_client_and_stops():
    _, clock, store, source, _ = await setup_ack()
    stop = asyncio.Event()
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path.endswith(f"/{CAMERA_ID}/configuration-ack")
        assert (
            request.headers["x-worker-generation-secret"]
            == source.binding.secret.get_secret_value()
        )
        assert body["workerGenerationId"] == source.binding.worker_generation_id
        assert body["streamSessionId"] == source.binding.stream_session_id
        assert body["bundleSha256"] == _configuration(1).bundle_sha256
        stop.set()
        return httpx.Response(
            200,
            json={
                "commandId": body["commandId"],
                "configurationVersion": 1,
                "outcome": "LOADED",
                "receivedAt": "2026-10-27T00:00:00.000Z",
                "replayed": False,
            },
        )

    client = BackendConfigurationClient(
        "https://backend.example", "synthetic-service-token", transport=httpx.MockTransport(handle)
    )
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
        require_bundle=True,
    )
    poller.enable_generation_ack(source)
    poller._last_success = clock.value
    await source.connect()
    await source.read_frame()
    try:
        await asyncio.wait_for(poller.run(stop), 1)
        assert len(requests) == 1
        await poller.run(stop)
        assert len(requests) == 1
    finally:
        await client.aclose()
        await source.close()


@pytest.mark.anyio
async def test_cancel_owned_poller_cancels_inflight_ack_without_extra_task():
    client, _, _, source, poller = await setup_ack()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client.acknowledge_configuration.side_effect = blocked
    await source.connect()
    await source.read_frame()
    task = asyncio.create_task(poller.run(asyncio.Event()))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert client.acknowledge_configuration.await_count == 1
    await source.close()


@pytest.mark.anyio
async def test_late_ack_for_replaced_bundle_cannot_confirm_new_context():
    client, _, store, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()

    async def replace(*args, **kwargs):
        await store.apply_bundle(_configuration(2))

    client.acknowledge_configuration.side_effect = replace
    try:
        await poller.acknowledge_loaded()
        assert poller._confirmed_ack is None
        assert poller._acknowledged_at is None
        client.acknowledge_configuration.side_effect = None
        await poller.acknowledge_loaded()
        assert poller._confirmed_ack.configuration_version == 2
    finally:
        await source.close()


@pytest.mark.anyio
async def test_ack_calls_are_serialized_and_coalesce_confirmed_same_bundle():
    client, _, _, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()
    entered, release = asyncio.Event(), asyncio.Event()
    active = 0
    max_active = 0

    async def send(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        await release.wait()
        active -= 1

    client.acknowledge_configuration.side_effect = send
    first = asyncio.create_task(poller.acknowledge_loaded())
    await entered.wait()
    second = asyncio.create_task(poller.acknowledge_loaded())
    await asyncio.sleep(0)
    release.set()
    try:
        await asyncio.gather(first, second)
        assert max_active == 1
        assert client.acknowledge_configuration.await_count == 1
    finally:
        await source.close()


@pytest.mark.anyio
async def test_terminal_ack_failure_clears_confirmed_state_and_metadata_freshness():
    client, clock, _, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()
    try:
        await poller.acknowledge_loaded()
        clock.value += 31
        client.acknowledge_configuration.side_effect = BackendConfigurationHttpError(403)
        with pytest.raises(BackendConfigurationHttpError):
            await poller.acknowledge_loaded()
        assert poller._confirmed_ack is None
        with pytest.raises(RuntimeError, match="never been confirmed"):
            poller.ensure_fresh()
    finally:
        await source.close()


@pytest.mark.anyio
async def test_uncertain_command_is_resolved_before_new_bundle_ack():
    client, _, store, source, poller = await setup_ack()
    await source.connect()
    await source.read_frame()
    client.acknowledge_configuration.side_effect = [
        BackendConfigurationTransientError(attempts=1),
        None,
        None,
    ]
    try:
        await poller.acknowledge_loaded()
        original = client.acknowledge_configuration.call_args.kwargs["acknowledgement"]
        await store.apply_bundle(_configuration(2))
        await poller.acknowledge_loaded()
        reconciled = client.acknowledge_configuration.call_args.kwargs["acknowledgement"]
        assert reconciled == original
        assert poller._confirmed_ack is None
        await poller.acknowledge_loaded()
        latest = client.acknowledge_configuration.call_args.kwargs["acknowledgement"]
        assert latest.configuration_version == 2
        assert latest.command_id != original.command_id
    finally:
        await source.close()
