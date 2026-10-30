import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from test_configuration_generation_ack import _configuration
from test_configuration_poller import CAMERA_ID, FakeClock
from test_worker_generation_binding import NOW, FakeSource, credential_payload

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_configuration_client import ConfigurationBundleResult
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.worker_generation import (
    GenerationBoundSource,
    WorkerGenerationCredential,
)


async def finite_replay(*, start=True):
    class FiniteSource(FakeSource):
        async def read_frame(self):
            if self.sequence:
                return None
            return (await super().read_frame()).model_copy(
                update={"camera_external_id": "CAM-GATE-01"}
            )

    clock = [NOW]
    source = GenerationBoundSource(
        FiniteSource(),
        WorkerGenerationCredential.model_validate(credential_payload()),
        clock=lambda: clock[0],
    )
    store = RegionConfigurationStore()
    client = AsyncMock()
    client.fetch_bundle.return_value = ConfigurationBundleResult(_configuration(1), '"v1"')
    poller = CameraConfigurationPoller(
        client=client,
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-GATE-01",
        monotonic_clock=FakeClock(),
        require_bundle=True,
    )
    poller.enable_generation_ack(source)
    await poller.refresh()
    stream = StreamWorker(
        config=StreamConfig(
            stream_id="synthetic",
            camera_external_id="CAM-GATE-01",
            source_url="fake://finite",
            is_live=False,
        ),
        source=source,
    )
    if start:
        await stream.start()
        await asyncio.wait_for(stream._loop_task, 2)
        assert not source.is_connected
    return clock, client, store, source, poller, stream


@pytest.mark.anyio
async def test_real_stream_eof_drains_final_frame_with_original_ack_context():
    _, client, store, source, poller, stream = await finite_replay()
    try:
        frame = await stream.get_frame()
        context = await poller.snapshot_delivery_context(
            await store.get_bundle("CAM-GATE-01"), frame, "REPLAY"
        )
        assert context is not None
        assert context.stream_session_id == source.binding.stream_session_id
        assert (
            context.receipt_command_id
            == client.acknowledge_configuration.call_args.kwargs["acknowledgement"].command_id
        )
        assert (
            await poller.snapshot_delivery_context(
                await store.get_bundle("CAM-GATE-01"), frame, "LIVE"
            )
            is None
        )
        with pytest.raises(RuntimeError, match="renewal"):
            await source.connect()
    finally:
        await stream.stop()


@pytest.mark.anyio
async def test_full_worker_run_keeps_final_finite_frame_and_delivers_bound_envelope(tmp_path):
    import json

    import httpx
    from test_processing_worker import REGION_ID, RecordingPipeline
    from test_processing_worker_bundle_policy import Detector

    from smartsite_ai.integrations.backend_client import BackendClient
    from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
    from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker

    _, _, store, source, poller, stream = await finite_replay(start=False)
    received = []

    def handle(request):
        envelope = json.loads(request.content)
        received.append(envelope)
        assert request.url.path == "/api/v1/integrations/ai/worker-events"
        assert envelope["context"]["sourceKind"] == "REPLAY"
        return httpx.Response(
            202,
            json={
                "eventId": envelope["event"]["eventId"],
                "status": "SKIPPED_NO_CANDIDATE",
                "alertIds": [],
            },
        )

    outbox = SqliteEventOutbox(tmp_path / "finite-worker.sqlite3")
    async with BackendClient(
        "http://backend.test", "synthetic-token", transport=httpx.MockTransport(handle)
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=Detector(),
            pipeline=RecordingPipeline(),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend, generation_credential=source.binding),
            configuration_guard=poller.ensure_fresh,
            delivery_context_provider=poller.snapshot_delivery_context,
        )
        result = await asyncio.wait_for(worker.run(), 3)
    assert result.frames_processed == 1
    assert result.events_enqueued == 1
    assert result.outbox.delivered == 1
    assert len(received) == 1
    assert not source.is_connected


@pytest.mark.anyio
async def test_eof_drain_never_renews_expired_generation_or_wrong_session():
    clock, _, store, _, poller, stream = await finite_replay()
    try:
        frame = await stream.get_frame()
        from uuid import uuid4

        wrong = frame.model_copy(update={"session_id": uuid4()})
        assert (
            await poller.snapshot_delivery_context(
                await store.get_bundle("CAM-GATE-01"), wrong, "REPLAY"
            )
            is None
        )
        clock[0] += timedelta(hours=1)
        with pytest.raises(RuntimeError, match="generation expired"):
            await poller.snapshot_delivery_context(
                await store.get_bundle("CAM-GATE-01"), frame, "REPLAY"
            )
    finally:
        await stream.stop()
