import asyncio
import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.live_latest_source import LiveLatestFrameSource
from smartsite_ai.media_preview import MediaPreviewHub


def frame(seq):
    return FrameEnvelope(
        stream_id="live",
        camera_external_id="cam",
        session_id=uuid4(),
        sequence_number=seq,
        captured_at=datetime.now(UTC),
        width=2,
        height=2,
        payload=b"\x00" * 12,
    )


def hub(**options):
    return MediaPreviewHub("cam", **options)


def test_media_publishes_while_inference_consumer_is_held_and_uses_one_source():
    async def run():
        media = hub(encoder=lambda value: b"\xff\xd8\xff\xd9")

        class Source:
            def __init__(self):
                self.items = asyncio.Queue()
                self.connects = 0

            async def connect(self):
                self.connects += 1

            async def close(self):
                pass

            async def read_frame(self):
                return await self.items.get()

        source = Source()
        reader = LiveLatestFrameSource(source)
        attach = getattr(reader, "set_media_sink", None)
        assert callable(attach), "Capture cannot fan out without waiting for inference"
        attach(media)
        await reader.connect()
        original = frame(7)
        source.items.put_nowait(original)
        for _ in range(50):
            await asyncio.sleep(0)
        raw = media.latest()
        assert raw is not None
        assert raw["sequenceNumber"] == "7"
        assert raw["capturedAt"] == original.captured_at_iso
        assert raw["sessionId"] == str(original.session_id)
        assert "detections" not in raw
        assert source.connects == 1
        source.items.put_nowait(None)
        await reader.close()
        assert media.latest() is None
        await media.close()

    asyncio.run(run())


def test_media_drops_pending_old_frames_and_clears_pixels():
    async def run():
        def encode(value):
            return b"\xff\xd8\xff\xd9"

        media = hub(encoder=encode)
        for seq in range(50):
            media.offer(frame(seq))
        for _ in range(50):
            await asyncio.sleep(0)
        assert media.latest()["sequenceNumber"] == "49"
        media.clear()
        assert media.latest() is None
        await media.close()
        assert media.latest() is None

    asyncio.run(run())


def test_clear_during_encoding_rejects_a_late_frame_from_the_old_source():
    async def run():
        entered, release = threading.Event(), threading.Event()

        def encode(value):
            entered.set()
            assert release.wait(2)
            return b"\xff\xd8\xff\xd9"

        media = hub(encoder=encode)
        try:
            media.offer(frame(1))
            assert await asyncio.to_thread(entered.wait, 2)
            media.clear()
            release.set()
            for _ in range(100):
                await asyncio.sleep(0)
            assert media.latest() is None
            assert media.frames_encoded == 0
        finally:
            release.set()
            await media.close()

    asyncio.run(run())


def test_media_encoding_error_and_stale_frame_never_leave_cached_pixels():
    async def run():
        tick = [0.0]
        media = hub(encoder=lambda value: b"\xff\xd8\xff\xd9", clock=lambda: tick[0])
        media.offer(frame(1))
        for _ in range(50):
            await asyncio.sleep(0)
        assert media.latest() is not None
        tick[0] = 2.1
        assert media.latest() is None
        await media.close()

    asyncio.run(run())


def test_wrong_camera_never_enters_encoder():
    async def run():
        calls = []
        media = MediaPreviewHub("other-camera", encoder=lambda value: calls.append(value))
        media.offer(frame(1))
        await asyncio.sleep(0)
        assert calls == []
        assert media.latest() is None
        await media.close()

    asyncio.run(run())


def test_encoder_failure_clears_previously_visible_pixels():
    async def run():
        first, failed = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()

        def encode(value):
            if value.sequence_number == 1:
                loop.call_soon_threadsafe(first.set)
                return b"\xff\xd8\xff\xd9"
            loop.call_soon_threadsafe(failed.set)
            raise ValueError("encoder unavailable")

        media = hub(encoder=encode)
        try:
            media.offer(frame(1))
            await asyncio.wait_for(first.wait(), 2)
            async with asyncio.timeout(2):
                while media.latest() is None:
                    await asyncio.sleep(0.001)
            assert media.latest() is not None
            media.offer(frame(2))
            await asyncio.wait_for(failed.wait(), 2)
            async with asyncio.timeout(2):
                while media.latest() is not None:
                    await asyncio.sleep(0.001)
            assert media.latest() is None
        finally:
            await media.close()

    asyncio.run(run())


def test_old_capture_is_not_made_fresh_by_recent_encoding():
    async def run():
        encoded = asyncio.Event()
        loop = asyncio.get_running_loop()

        def encode(value):
            loop.call_soon_threadsafe(encoded.set)
            return b"\xff\xd8\xff\xd9"

        media = hub(encoder=encode)
        original = frame(1)
        stale = FrameEnvelope.model_validate(
            {**original.model_dump(), "captured_at": datetime.now(UTC) - timedelta(seconds=10)}
        )
        try:
            media.offer(stale)
            await asyncio.wait_for(encoded.wait(), 2)
            for _ in range(100):
                await asyncio.sleep(0)
            assert media.latest() is None
        finally:
            await media.close()

    asyncio.run(run())


def test_default_encoder_produces_bounded_jpeg_with_actual_dimensions():
    import io

    from PIL import Image

    from smartsite_ai.media_preview import encode_media_frame

    encoded, width, height = encode_media_frame(frame(1))
    assert len(encoded) <= 131072
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.format == "JPEG"
        assert image.size == (width, height) == (2, 2)
