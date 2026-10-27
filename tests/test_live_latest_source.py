import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.live_latest_source import LiveLatestFrameSource
from smartsite_ai.ingestion.source import SourceReadError


class Source:
    def __init__(self):
        self.items = asyncio.Queue()
        self.reading = asyncio.Event()
        self.closed = 0

    async def connect(self):
        pass

    async def read_frame(self):
        self.reading.set()
        value = await self.items.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closed += 1


def frame(seq, session=None):
    return FrameEnvelope(
        stream_id="live",
        camera_external_id="cam",
        session_id=session or uuid4(),
        sequence_number=seq,
        captured_at=datetime.now(UTC),
        width=1,
        height=1,
        payload=b"\x00\x00\x00",
    )


async def spin():
    for _ in range(10):
        await asyncio.sleep(0)


def test_slow_consumer_receives_latest_original_frame_without_retimestamping():
    async def scenario():
        source = Source()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        frames = [frame(i) for i in range(100)]
        for value in frames:
            source.items.put_nowait(value)
        for _ in range(110):
            await asyncio.sleep(0)
        assert reader.frames_read == 100
        assert reader.frames_replaced == 99
        assert await reader.read_frame() is frames[-1]
        source.items.put_nowait(None)
        await reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", [None, SourceReadError("rtsp://private-secret")])
def test_terminal_read_discards_queued_pixels_and_never_exposes_source_error(terminal):
    async def scenario():
        source = Source()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        source.items.put_nowait(frame(0))
        source.items.put_nowait(terminal)
        await spin()
        assert not reader.is_connected
        if terminal is None:
            assert await reader.read_frame() is None
        else:
            with pytest.raises(SourceReadError, match="^Live frame reader is unavailable$"):
                await reader.read_frame()
        await reader.close()

    asyncio.run(scenario())


def test_cancelled_consumer_cannot_steal_next_frame():
    async def scenario():
        source = Source()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        waiting = asyncio.create_task(reader.read_frame())
        await spin()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        fresh = frame(1)
        source.items.put_nowait(fresh)
        assert await reader.read_frame() is fresh
        source.items.put_nowait(None)
        await reader.close()

    asyncio.run(scenario())


def test_close_wakes_consumer_but_waits_for_native_read_before_release():
    async def scenario():
        source = Source()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        await source.reading.wait()
        waiting = asyncio.create_task(reader.read_frame())
        closing = asyncio.create_task(reader.close())
        await spin()
        with pytest.raises(SourceReadError):
            await waiting
        assert not closing.done()
        assert source.closed == 1  # initial connect close only
        source.items.put_nowait(frame(1))
        await closing
        assert source.closed == 2

    asyncio.run(scenario())


def test_reconnect_does_not_return_old_session_frame():
    async def scenario():
        source = Source()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        source.items.put_nowait(frame(50))
        source.items.put_nowait(None)
        await spin()
        await reader.connect()
        fresh = frame(0)
        source.items.put_nowait(fresh)
        assert await reader.read_frame() is fresh
        source.items.put_nowait(None)
        await reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled,expected", [(False, [0, 1]), (True, [0, 3])])
def test_actual_realtime_pipeline_skips_backlog_only_when_explicitly_enabled(
    monkeypatch, enabled, expected
):
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import WebSocketDisconnect

    from smartsite_ai import realtime
    from smartsite_ai.config import Settings
    from smartsite_ai.inference.models import DetectionBatch

    async def scenario():
        source = Source()
        session = uuid4()
        source.items.put_nowait(frame(0, session))
        detector, runner, pipeline = MagicMock(), MagicMock(), MagicMock()
        reads = 0

        async def detect(envelope):
            nonlocal reads
            reads += 1
            if reads == 1:
                for i in (1, 2, 3):
                    source.items.put_nowait(frame(i, session))
                await spin()  # camera keeps reading while fake inference works
            return DetectionBatch.from_frame(
                envelope,
                model_artifact_id="fake",
                model_version="test",
                model_sha256="a" * 64,
                detections=(),
            )

        detector.detect = detect
        pipeline.observe.return_value = None
        pipeline.occupied_zone_track_regions.return_value = frozenset()
        monkeypatch.setattr(realtime, "OpenCvFrameSource", lambda config: source)
        monkeypatch.setattr(
            realtime,
            "_load_realtime_stack",
            MagicMock(return_value=(detector, runner, pipeline, None)),
        )
        sent = []

        async def send(value):
            if value["type"] == "frame":
                sent.append(int(value["sequenceNumber"]))
                if len(sent) == 2:
                    source.items.put_nowait(None)
                    raise WebSocketDisconnect()

        socket = MagicMock(accept=AsyncMock(), close=AsyncMock(), send_json=send)
        media = MagicMock()
        await realtime.stream_realtime(
            socket,
            Settings(_env_file=None, realtime_source="0", realtime_live_latest_frame=enabled),
            media_hub=media,
        )
        if enabled:
            assert media.offer.call_count >= 4
            assert media.clear.called
        else:
            media.offer.assert_not_called()
        assert sent == expected
        runner.close.assert_called_once()

    asyncio.run(scenario())
