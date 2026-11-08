"""One durable ingestion source must serve display independently of inference."""

import asyncio
import json
from datetime import timedelta

import httpx
import pytest
from configuration_capture_fakes import NOW, FakeSource, frame, until
from fastapi import FastAPI
from pydantic import SecretStr
from test_processing_worker import REGION_ID, FakeDetector, RecordingPipeline, _configuration

from smartsite_ai.configuration_media_api import configuration_media_router
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import SourceReadError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.media_preview import MediaPreviewHub
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker

TOKEN = "synthetic-durable-worker-media-service-token"
PATH = "/v1/realtime/video-stream?cameraExternalId=CAM-GATE-01"


def stream_config(**changes):
    return StreamConfig(
        stream_id="configuration-stream",
        camera_external_id="CAM-GATE-01",
        source_url="test://camera",
        **changes,
    )


def test_raw_advances_while_detector_waits_and_viewer_cancel_preserves_delivery(tmp_path):
    async def scenario():
        source = FakeSource()
        hub = MediaPreviewHub("CAM-GATE-01", utc_clock=lambda: NOW)
        stream = StreamWorker(stream_config(), source, frame_observer=hub)
        app = FastAPI()
        app.include_router(configuration_media_router(stream, hub, SecretStr(TOKEN)))
        entered, release = asyncio.Event(), asyncio.Event()
        delivered = []

        class SlowDetector(FakeDetector):
            async def detect(self, value):
                entered.set()
                await release.wait()
                return await super().detect(value)

        def handler(request):
            item = json.loads(request.content)
            delivered.append(item)
            return httpx.Response(
                202, json={"eventId": item["eventId"], "status": "PROCESSED", "alertIds": []}
            )

        store = RegionConfigurationStore()
        await store.apply(_configuration())
        outbox = SqliteEventOutbox((tmp_path / "outbox.sqlite3").resolve())
        async with BackendClient(
            "http://backend:3000", "synthetic", transport=httpx.MockTransport(handler)
        ) as backend:
            worker = HeadlessCameraProcessingWorker(
                stream=stream,
                detector=SlowDetector(),
                pipeline=RecordingPipeline(),
                ppe_region_id=REGION_ID,
                configurations=store,
                outbox=outbox,
                dispatcher=OutboxDispatcher(outbox, backend),
                delivery_interval_seconds=0.01,
            )
            task = asyncio.create_task(worker.run())
            viewer = None
            seen = asyncio.Event()

            async def observed_app(scope, receive, send):
                async def observed_send(message):
                    await send(message)
                    if message["type"] == "http.response.body" and message.get("body"):
                        assert json.loads(message["body"])["sequenceNumber"] == "1"
                        seen.set()

                await app(scope, receive, observed_send)

            try:
                await source.connected.wait()
                source.items.put_nowait(frame(camera_external_id="CAM-GATE-01"))
                await asyncio.wait_for(entered.wait(), 2)
                second = frame(
                    1, camera_external_id="CAM-GATE-01", captured_at=NOW + timedelta(seconds=0.1)
                )
                source.items.put_nowait(second)
                await until(
                    lambda: hub.latest() is not None and hub.latest()["sequenceNumber"] == "1"
                )
                assert delivered == []
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(observed_app), base_url="http://test"
                ) as client:
                    viewer = asyncio.create_task(
                        client.get(PATH, headers={"Authorization": f"Bearer {TOKEN}"})
                    )
                    await asyncio.wait_for(seen.wait(), 2)
                    viewer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await viewer
                    assert source.connects == 1 and source.closes == 0
                    release.set()

                    def delivered_or_failure():
                        if task.done():
                            task.result()
                        return len(delivered) == 2

                    await until(delivered_or_failure)
                    async with asyncio.timeout(2):
                        while (await outbox.counts()).delivered != 2:
                            await asyncio.sleep(0.001)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (await outbox.counts()).delivered == 2
                assert [item["capturedAt"] for item in delivered] == [
                    frame().captured_at_iso,
                    second.captured_at_iso,
                ]
                assert hub.latest() is None and not stream.observer_available
            finally:
                release.set()
                if viewer is not None and not viewer.done():
                    viewer.cancel()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await stream.stop()
                await hub.close()
        assert source.connects == source.closes == 1

    asyncio.run(scenario())


def test_validated_frames_reach_display_before_inference_sampling(tmp_path):
    async def scenario():
        source = FakeSource()
        hub = MediaPreviewHub("CAM-GATE-01", utc_clock=lambda: NOW)
        stream = StreamWorker(stream_config(target_fps=1), source, frame_observer=hub)
        await stream.start()
        try:
            source.items.put_nowait(frame(camera_external_id="wrong-camera"))
            source.items.put_nowait(frame(camera_external_id="CAM-GATE-01"))
            source.items.put_nowait(
                frame(1, camera_external_id="CAM-GATE-01", captured_at=NOW + timedelta(seconds=0.1))
            )
            await until(lambda: stream.sampled_out_frames == 1)
            await until(lambda: hub.latest() is not None and hub.latest()["sequenceNumber"] == "1")
            assert stream.integrity_errors == 1
            assert (await stream.get_frame()).sequence_number == 0
            assert stream.queue.qsize() == 0
        finally:
            await stream.stop()
            await hub.close()

    asyncio.run(scenario())


def test_replay_cannot_enable_live_raw_observer():
    with pytest.raises(ValueError, match="LIVE"):
        StreamWorker(stream_config(is_live=False), FakeSource(), frame_observer=object())


def test_durable_preview_request_cannot_start_its_source():
    async def scenario():
        source = FakeSource()
        hub = MediaPreviewHub("CAM-GATE-01", utc_clock=lambda: NOW)
        stream = StreamWorker(stream_config(), source, frame_observer=hub)
        app = FastAPI()
        app.include_router(configuration_media_router(stream, hub, SecretStr(TOKEN)))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.get(PATH, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 503
        assert source.connects == source.reads == source.closes == 0
        assert not stream.observer_available
        await hub.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("ending", [None, SourceReadError("synthetic secret")])
def test_source_failure_revokes_raw_display_before_terminal_cleanup(ending):
    async def scenario():
        source = FakeSource()
        hub = MediaPreviewHub("CAM-GATE-01", utc_clock=lambda: NOW)
        stream = StreamWorker(stream_config(max_consecutive_failures=1), source, frame_observer=hub)
        await stream.start()
        try:
            source.items.put_nowait(frame(camera_external_id="CAM-GATE-01"))
            await until(lambda: hub.latest() is not None)
            source.items.put_nowait(ending)
            await until(lambda: stream.state == StreamState.ERROR)
            assert hub.latest() is None and not stream.observer_available
            assert source.connects == source.closes == 1
            assert "secret" not in (stream.last_error or "")
        finally:
            await stream.stop()
            await hub.close()

    asyncio.run(scenario())


def test_broken_observer_cannot_reconnect_or_skip_source_cleanup():
    class BrokenObserver:
        def offer(self, value):
            raise RuntimeError("synthetic credential")

        def clear(self):
            raise RuntimeError("synthetic credential")

    async def scenario():
        source = FakeSource()
        stream = StreamWorker(stream_config(), source, frame_observer=BrokenObserver())
        await stream.start()
        try:
            source.items.put_nowait(frame(camera_external_id="CAM-GATE-01"))
            assert (await asyncio.wait_for(stream.get_frame(), 2)).sequence_number == 0
            assert stream.state == StreamState.STREAMING
            assert not stream.observer_available
            assert stream.last_error is None and stream.reconnect_attempts == 0
        finally:
            await stream.stop()
        assert source.connects == source.closes == 1

    asyncio.run(scenario())
