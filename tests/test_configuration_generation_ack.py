import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_configuration_poller import CAMERA_ID, FakeClock, _configuration
from test_worker_generation_binding import NOW, FakeSource, credential_payload

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
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


async def setup_ack():
    client = AsyncMock()
    clock = FakeClock()
    store = RegionConfigurationStore()
    await store.apply_bundle(_configuration(1))
    source = GenerationBoundSource(
        FakeSource(),
        WorkerGenerationCredential.model_validate(credential_payload()),
        clock=lambda: NOW,
    )
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=clock,
    )
    poller.enable_generation_ack(source)
    client.fetch_bundle.return_value = ConfigurationBundleResult(_configuration(1), '"v1"')
    await poller.refresh()
    return client, clock, store, source, poller


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
