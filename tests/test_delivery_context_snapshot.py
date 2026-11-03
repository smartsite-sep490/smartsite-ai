import pytest
from test_configuration_generation_ack import _configuration, setup_ack

from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationTransientError,
)


@pytest.mark.anyio
async def test_first_replay_frame_obtains_ack_without_waiting_for_background_poll():
    client, _, store, source, poller = await setup_ack()
    await source.connect()
    frame = (await source.read_frame()).model_copy(update={"camera_external_id": "CAM-GATE-01"})
    context = await poller.snapshot_delivery_context(
        await store.get_bundle("CAM-GATE-01"), frame, "REPLAY"
    )
    assert context is not None
    assert (
        context.receipt_command_id
        == client.acknowledge_configuration.call_args.kwargs["acknowledgement"].command_id
    )
    await source.close()


@pytest.mark.anyio
async def test_context_requires_confirmed_exact_bundle_and_bound_frame():
    client, _, store, source, poller = await setup_ack()
    snapshot = getattr(poller, "snapshot_delivery_context", None)
    assert callable(snapshot)
    bundle = await store.get_bundle("CAM-GATE-01")
    await source.connect()
    frame = (await source.read_frame()).model_copy(update={"camera_external_id": "CAM-GATE-01"})
    client.acknowledge_configuration.side_effect = BackendConfigurationTransientError(attempts=1)
    assert await snapshot(bundle, frame, "LIVE") is None
    client.acknowledge_configuration.side_effect = None
    await poller.acknowledge_loaded()
    context = await snapshot(bundle, frame, "LIVE")
    sent = client.acknowledge_configuration.call_args.kwargs["acknowledgement"]
    assert context.receipt_command_id == sent.command_id
    assert context.worker_generation_id == source.binding.worker_generation_id
    assert context.stream_session_id == str(frame.session_id)
    assert context.bundle_sha256 == bundle.bundle_sha256
    assert source.binding.secret.get_secret_value() not in context.model_dump_json()
    wrong = frame.model_copy(update={"camera_external_id": "OTHER"})
    assert await snapshot(bundle, wrong, "LIVE") is None
    await store.apply_bundle(_configuration(2))
    assert await snapshot(bundle, frame, "LIVE") is None
    assert (
        await snapshot(await store.get_bundle("CAM-GATE-01"), frame, "LIVE")
    ).configuration_version == 2
    await source.close()


@pytest.mark.anyio
async def test_uncertain_ack_preserves_only_the_confirmed_command():
    client, clock, store, source, poller = await setup_ack()
    snapshot = getattr(poller, "snapshot_delivery_context", None)
    assert callable(snapshot)
    await source.connect()
    frame = (await source.read_frame()).model_copy(update={"camera_external_id": "CAM-GATE-01"})
    bundle = await store.get_bundle("CAM-GATE-01")
    client.acknowledge_configuration.side_effect = BackendConfigurationTransientError(attempts=1)
    await poller.acknowledge_loaded()
    assert await snapshot(bundle, frame, "LIVE") is None
    client.acknowledge_configuration.side_effect = None
    await poller.acknowledge_loaded()
    original = await snapshot(bundle, frame, "LIVE")
    assert original is not None
    clock.value += 30
    client.acknowledge_configuration.side_effect = BackendConfigurationTransientError(attempts=1)
    await poller.acknowledge_loaded()
    assert await snapshot(bundle, frame, "LIVE") == original
    await source.close()
    assert await snapshot(bundle, frame, "LIVE") is None
