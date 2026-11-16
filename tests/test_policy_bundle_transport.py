import json
from uuid import UUID

import httpx
import pytest
from test_policy_configuration_bundle import content, packet
from test_worker_generation_binding import NOW, FakeSource, credential_payload

from smartsite_ai.core.region_configuration_store import (
    RegionConfigurationApplyOutcome,
    RegionConfigurationStore,
)
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    BackendConfigurationResponseError,
)
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.worker_generation import (
    GenerationBoundSource,
    WorkerGenerationCredential,
)


@pytest.mark.anyio
async def test_v2_poll_installs_bundle_and_rejects_etag_from_different_payload():
    value = content()
    body = packet(value)
    digest = json.loads(body)["bundleSha256"]
    etag = f'"sha256:{digest}"'
    requests = []
    binding = WorkerGenerationCredential.model_validate(
        {**credential_payload(), "cameraId": value["cameraId"]}
    )
    source = GenerationBoundSource(FakeSource(), binding, clock=lambda: NOW)
    acknowledgements = []

    def backend(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path.endswith("/configuration-ack")
            assert (
                request.headers["x-worker-generation-secret"] == binding.secret.get_secret_value()
            )
            ack = json.loads(request.content)
            acknowledgements.append(ack)
            assert ack["bundleSha256"] == digest
            assert ack["configurationVersion"] == value["configurationVersion"]
            assert ack["viewRevision"] == value["viewRevision"]
            assert ack["workerGenerationId"] == binding.worker_generation_id
            assert ack["streamSessionId"] == binding.stream_session_id
            return httpx.Response(
                200,
                json={
                    "commandId": ack["commandId"],
                    "configurationVersion": ack["configurationVersion"],
                    "outcome": "LOADED",
                    "receivedAt": "2026-10-27T00:00:00.000Z",
                    "replayed": False,
                },
            )
        requests.append(request)
        assert request.url.path.endswith("/configuration-bundle/v2")
        if len(requests) == 2:
            assert request.headers["if-none-match"] == etag
            return httpx.Response(304, headers={"ETag": etag})
        response_tag = etag if len(requests) == 1 else '"sha256:' + "0" * 64 + '"'
        return httpx.Response(
            200, content=body, headers={"ETag": response_tag, "Content-Type": "application/json"}
        )

    async with BackendConfigurationClient(
        "https://backend.example", "test", policy_v2=True, transport=httpx.MockTransport(backend)
    ) as client:
        store = RegionConfigurationStore(allow_policy_v2=True)
        poller = CameraConfigurationPoller(
            client=client,
            store=store,
            camera_id=UUID(value["cameraId"]),
            camera_external_id=value["cameraExternalId"],
            require_bundle=True,
        )
        poller.enable_generation_ack(source)
        await source.connect()
        frame = await source.read_frame()
        assert await poller.refresh() is RegionConfigurationApplyOutcome.APPLIED
        snapshot = await store.get_bundle(value["cameraExternalId"])
        assert snapshot is not None and frame is not None
        frame = frame.model_copy(update={"camera_external_id": value["cameraExternalId"]})
        context = await poller.snapshot_delivery_context(snapshot, frame, "LIVE")
        assert context is not None
        assert context.receipt_command_id == acknowledgements[0]["commandId"]
        assert context.bundle_sha256 == digest
        # Current Main supports explicit replay contexts and finite-queue draining.
        replay = await poller.snapshot_delivery_context(snapshot, frame, "REPLAY")
        assert replay is not None and replay.source_kind == "REPLAY"
        assert replay.receipt_command_id == context.receipt_command_id
        assert len(acknowledgements) == 1
        assert await poller.refresh() is RegionConfigurationApplyOutcome.IDEMPOTENT
        with pytest.raises(BackendConfigurationResponseError, match="ETag"):
            await poller.refresh()
        assert poller.etag is None
        snapshot = await store.get_active_configuration(value["cameraExternalId"])
        assert snapshot is not None
        assert snapshot.bundle_sha256 == digest


@pytest.mark.anyio
async def test_tracked_scope_pinning_never_installs_or_acknowledges_a_changed_bundle():
    import copy
    from unittest.mock import AsyncMock

    from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
    from smartsite_ai.integrations.backend_configuration_client import ConfigurationBundleResult
    from smartsite_ai.integrations.configuration_poller import ConfigurationStaleError

    value = content()
    original = CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))
    changed_value = copy.deepcopy(value)
    changed_value["configurationVersion"] += 1
    changed_value["cameraPpePolicy"] = {"mode": "DISABLED", "reason": "Approved later policy"}
    changed = CameraPolicyConfigurationBundle.from_wire_bytes(packet(changed_value))
    client = AsyncMock()
    client.fetch_bundle.side_effect = [
        ConfigurationBundleResult(original, '"first"'),
        ConfigurationBundleResult(changed, '"second"'),
        ConfigurationBundleResult(changed, '"second"'),
    ]
    store = RegionConfigurationStore(allow_policy_v2=True)
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=UUID(value["cameraId"]),
        camera_external_id=value["cameraExternalId"],
        require_bundle=True,
        pin_bundle_scope=True,
    )
    binding = WorkerGenerationCredential.model_validate(
        {**credential_payload(), "cameraId": value["cameraId"]}
    )
    source = GenerationBoundSource(FakeSource(), binding, clock=lambda: NOW)
    poller.enable_generation_ack(source)
    await source.connect()
    await source.read_frame()
    await poller.refresh()
    await poller.acknowledge_loaded()
    assert client.acknowledge_configuration.await_count == 1
    for _ in range(2):
        with pytest.raises(RuntimeError, match="scope transition") as error:
            await poller.refresh()
        assert type(error.value).__name__ == "TrackedScopeTransitionRequired"
        assert await store.get_bundle(value["cameraExternalId"]) is original
        with pytest.raises(ConfigurationStaleError):
            poller.ensure_fresh()
    assert client.acknowledge_configuration.await_count == 1
    await source.close()
