"""Purpose JPEG and display share one ingestion source, not one relabeled image."""

import asyncio
import base64
import hashlib
import threading
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from configuration_capture_fakes import NOW, FakeSource, frame, until
from fastapi import FastAPI
from pydantic import SecretStr

from smartsite_ai.configuration_image import ConfigurationJpeg
from smartsite_ai.configuration_image_api import configuration_image_router
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.media_preview import MediaPreviewHub

TOKEN = "synthetic-durable-configuration-image-token"
PATH = "/v1/realtime/configuration-image?cameraExternalId=camera-a"


def test_one_capture_supplies_exact_purpose_image_and_raw_display():
    async def scenario():
        from smartsite_ai.durable_preview import DurablePreviewFrames

        now = [NOW]
        frames = DurablePreviewFrames("camera-a", clock=lambda: now[0])
        source = FakeSource()
        stream = StreamWorker(
            StreamConfig(stream_id=source.source_id, camera_external_id="camera-a", source_url="0"),
            source,
            frame_observer=frames,
        )
        app = FastAPI()
        app.include_router(configuration_image_router(frames.configuration, SecretStr(TOKEN)))
        headers = {"Authorization": f"Bearer {TOKEN}"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (await client.get(PATH, headers=headers)).status_code == 503
            assert source.connects == source.reads == 0
            await stream.start()
            value = frame(width=1280, height=720, payload=b"\x00\x20\x40" * 1280 * 720)
            source.items.put_nowait(value)
            try:
                await until(
                    lambda: (
                        frames.configuration.latest() is not None and frames.latest() is not None
                    )
                )
                response = await client.get(PATH, headers=headers)
                assert response.status_code == 200
                data = response.json()
                snapshot = data["frame"]
                assert data["purpose"] == "REGION_CONFIGURATION" and data["sourceKind"] == "LIVE"
                assert snapshot["sessionId"] == str(value.session_id)
                assert (
                    snapshot["sequenceNumber"] == "0"
                    and snapshot["capturedAt"] == value.captured_at_iso
                )
                assert (snapshot["width"], snapshot["height"]) == (1280, 720)
                assert frames.latest()["width"] == 640
                jpeg = base64.b64decode(snapshot["imageDataUrl"].split(",", 1)[1])
                assert hashlib.sha256(jpeg).hexdigest() == snapshot["imageSha256"]
                assert 0 < len(jpeg) <= 1024 * 1024
                count = frames.configuration.frames_encoded
                for _ in range(3):
                    assert (await client.get(PATH, headers=headers)).json() == data
                assert frames.configuration.frames_encoded == count
                assert (await client.get(PATH)).status_code == 401
                assert (
                    await client.get(PATH.replace("camera-a", "other"), headers=headers)
                ).status_code == 404
                now[0] += timedelta(seconds=3)
                assert (await client.get(PATH, headers=headers)).status_code == 503
                await stream.stop()
                now[0] = NOW
                assert (await client.get(PATH, headers=headers)).status_code == 503
            finally:
                await stream.stop()
                await frames.close()
        assert source.connects == source.closes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["raw", "configuration"])
def test_encoder_close_drains_thread_despite_repeated_cancellation(kind):
    async def scenario():
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()

        def encode(value):
            entered.set()
            assert release.wait(2)
            exited.set()
            if kind == "configuration":
                return ConfigurationJpeg(b"\xff\xd8\xff\xd9", 64, 64)
            return b"\xff\xd8\xff\xd9"

        if kind == "configuration":
            from smartsite_ai.configuration_image_observer import ConfigurationImageObserver

            observer = ConfigurationImageObserver("camera-a", clock=lambda: NOW, encoder=encode)
        else:
            observer = MediaPreviewHub("camera-a", utc_clock=lambda: NOW, encoder=encode)
        observer.offer(frame())
        closer = None
        try:
            await until(entered.is_set)
            closer = asyncio.create_task(observer.close())
            done, _ = await asyncio.wait({closer}, timeout=0.02)
            assert not done and not exited.is_set()
            assert observer.latest() is None
            closer.cancel()
            await asyncio.sleep(0)
            closer.cancel()
            assert not closer.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closer, 2)
            assert exited.is_set() and observer.latest() is None
        finally:
            release.set()
            if closer is not None:
                await asyncio.gather(closer, return_exceptions=True)
            await observer.close()

    asyncio.run(scenario())


def test_configuration_late_result_cannot_restore_old_session():
    async def scenario():
        from smartsite_ai.configuration_image_observer import ConfigurationImageObserver

        entered, release = threading.Event(), threading.Event()
        sessions = []

        def encode(value):
            sessions.append(value.session_id)
            if len(sessions) == 1:
                entered.set()
                assert release.wait(2)
            return ConfigurationJpeg(b"\xff\xd8\xff\xd9", 64, 64)

        observer = ConfigurationImageObserver("camera-a", clock=lambda: NOW, encoder=encode)
        new = frame(session_id=uuid4())
        try:
            observer.offer(frame())
            await until(entered.is_set)
            observer.clear()
            observer.offer(new)
            release.set()
            await until(lambda: observer.latest() is not None)
            assert observer.latest()["frame"]["sessionId"] == str(new.session_id)
            assert observer.frames_encoded == 1
        finally:
            release.set()
            await observer.close()

    asyncio.run(scenario())


def test_configuration_replaces_pending_frames_instead_of_encoding_a_history():
    async def scenario():
        from smartsite_ai.configuration_image_observer import ConfigurationImageObserver

        entered, release = threading.Event(), threading.Event()
        sequences = []

        def encode(value):
            sequences.append(value.sequence_number)
            if len(sequences) == 1:
                entered.set()
                assert release.wait(2)
            return ConfigurationJpeg(b"\xff\xd8\xff\xd9", 64, 64)

        observer = ConfigurationImageObserver("camera-a", clock=lambda: NOW, encoder=encode)
        try:
            observer.offer(frame())
            await until(entered.is_set)
            for number in range(1, 50):
                observer.offer(frame(number))
            release.set()
            await until(
                lambda: (
                    observer.latest() is not None
                    and observer.latest()["frame"]["sequenceNumber"] == "49"
                )
            )
            assert sequences == [0, 49]
        finally:
            release.set()
            await observer.close()

    asyncio.run(scenario())


def test_configuration_encoding_failure_does_not_disable_raw_display():
    async def scenario():
        from smartsite_ai.configuration_image_observer import ConfigurationImageObserver
        from smartsite_ai.durable_preview import DurablePreviewFrames

        calls = []

        def broken(value):
            calls.append(value)
            raise RuntimeError("synthetic credential")

        observers = DurablePreviewFrames("camera-a", clock=lambda: NOW)
        observers.configuration = ConfigurationImageObserver(
            "camera-a", clock=lambda: NOW, encoder=broken
        )
        try:
            observers.offer(frame())
            await until(lambda: observers.latest() is not None and len(calls) == 1)
            observers.offer(frame(1))
            await until(lambda: observers.latest()["sequenceNumber"] == "1")
            assert observers.configuration.latest() is None and len(calls) == 1
        finally:
            await observers.close()

    asyncio.run(scenario())


def test_configuration_runtime_reports_pending_while_raw_encoder_drains(tmp_path):
    from test_configuration_media_preview import runtime_pair

    from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease

    async def scenario():
        entered, release = threading.Event(), threading.Event()
        runtime, source, _ = runtime_pair(tmp_path, enabled=True)

        def encode(value):
            entered.set()
            assert release.wait(2)
            return b"\xff\xd8\xff\xd9"

        runtime._media_hub._encoder = encode
        await runtime.start()
        source.items.put_nowait(frame())
        competitor = ConfigurationProcessLease(
            tmp_path, UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")
        )
        try:
            await until(entered.is_set)
            source.items.put_nowait(None)
            assert not await asyncio.wait_for(runtime.stop(), 0.3)
            with pytest.raises(RuntimeError, match="ownership"):
                competitor.acquire()
            release.set()
            assert await asyncio.wait_for(runtime.wait_for_release(), 2)
            competitor.acquire()
            competitor.release()
        finally:
            release.set()
            source.items.put_nowait(None)
            await runtime.stop()
            await runtime.wait_for_release()

    asyncio.run(scenario())
