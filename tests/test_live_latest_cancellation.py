"""Deterministic cancellation regression tests for LiveLatestFrameSource (AI-A04)."""

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.live_latest_source import LiveLatestFrameSource
from smartsite_ai.ingestion.source import SourceReadError


class FakeMediaSink:
    def __init__(self) -> None:
        self.offered: list[FrameEnvelope] = []
        self.current: FrameEnvelope | None = None
        self.cleared = 0

    def offer(self, frame: FrameEnvelope) -> None:
        self.offered.append(frame)
        self.current = frame

    def clear(self) -> None:
        self.current = None
        self.cleared += 1


class FakeSource:
    def __init__(self) -> None:
        self.items: asyncio.Queue[FrameEnvelope | BaseException | None] = asyncio.Queue()
        self.reading = asyncio.Event()
        self.in_flight_reads = 0
        self.closed = 0
        self.closed_during_read = False
        self._connected = False

    @property
    def source_id(self) -> str:
        return "fake-live"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def read_frame(self) -> FrameEnvelope | None:
        self.in_flight_reads += 1
        self.reading.set()
        try:
            value = await self.items.get()
            if isinstance(value, BaseException):
                raise value
            return value
        finally:
            self.in_flight_reads -= 1
            if self.in_flight_reads == 0:
                self.reading.clear()

    async def close(self) -> None:
        if self.in_flight_reads > 0:
            self.closed_during_read = True
        self._connected = False
        self.closed += 1


def make_frame(seq: int, session: UUID | None = None) -> FrameEnvelope:
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


async def spin(ticks: int = 10) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def test_internal_cancelled_error_makes_reader_unavailable_clears_state_and_propagates() -> None:
    async def scenario() -> None:
        source = FakeSource()
        sink = FakeMediaSink()
        reader = LiveLatestFrameSource(source)
        reader.set_media_sink(sink)
        first = make_frame(0)
        source.items.put_nowait(first)
        source.items.put_nowait(asyncio.CancelledError("internal read cancelled"))

        await reader.connect()
        reader_task = reader._task
        assert reader_task is not None
        try:
            await spin()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(reader_task, timeout=0.5)
            assert reader_task.cancelled()
            assert reader.frames_read == 1
            assert sink.offered == [first]

            assert not reader.is_connected
            assert reader._latest is None
            assert sink.current is None
            assert sink.cleared >= 1

            with pytest.raises(SourceReadError, match="^Live frame reader is unavailable$"):
                await asyncio.wait_for(reader.read_frame(), timeout=0.5)
        finally:
            source.items.put_nowait(None)
            with suppress(BaseException):
                await asyncio.wait_for(reader_task, timeout=0.5)
            with suppress(BaseException):
                await asyncio.wait_for(reader.close(), timeout=0.5)

    asyncio.run(scenario())


def test_uncancelled_close_and_reconnect_recover_after_internal_reader_cancellation() -> None:
    async def scenario() -> None:
        source = FakeSource()
        sink = FakeMediaSink()
        reader = LiveLatestFrameSource(source)
        reader.set_media_sink(sink)
        stale = make_frame(0)
        source.items.put_nowait(stale)
        source.items.put_nowait(asyncio.CancelledError("internal read cancelled"))

        await reader.connect()
        first_task = reader._task
        assert first_task is not None
        try:
            await spin()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first_task, timeout=0.5)
            assert first_task.cancelled()
            assert not reader.is_connected
            with pytest.raises(SourceReadError, match="^Live frame reader is unavailable$"):
                await asyncio.wait_for(reader.read_frame(), timeout=0.5)

            await asyncio.wait_for(reader.close(), timeout=0.5)
            await asyncio.wait_for(reader.connect(), timeout=0.5)
            assert reader.is_connected

            fresh = make_frame(1)
            source.items.put_nowait(fresh)
            await spin()
            received = await asyncio.wait_for(reader.read_frame(), timeout=0.5)
            assert received is fresh
            assert received is not stale
            assert sink.current is fresh
        finally:
            source.items.put_nowait(None)
            with suppress(BaseException):
                await asyncio.wait_for(first_task, timeout=0.5)
            if reader._task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(reader._task, timeout=0.5)
            with suppress(BaseException):
                await asyncio.wait_for(reader.close(), timeout=0.5)

    asyncio.run(scenario())


def test_close_cancelled_twice_waits_for_native_read_then_releases_once_and_propagates() -> None:
    async def scenario() -> None:
        source = FakeSource()
        reader = LiveLatestFrameSource(source)
        await reader.connect()
        reader_task = reader._task
        assert reader_task is not None
        closing: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(source.reading.wait(), timeout=0.5)
            assert source.in_flight_reads == 1
            closed_before_close = source.closed

            closing = asyncio.create_task(reader.close())
            await spin()
            assert not closing.done()
            assert not source.closed_during_read
            assert source.closed == closed_before_close

            closing.cancel()
            await spin()
            assert not closing.done()
            assert not source.closed_during_read
            assert source.closed == closed_before_close

            closing.cancel()
            await spin()
            assert not closing.done()
            assert not source.closed_during_read
            assert source.closed == closed_before_close

            source.items.put_nowait(make_frame(1))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, timeout=0.5)

            assert reader_task.done()
            assert not source.closed_during_read
            assert source.closed == closed_before_close + 1
        finally:
            source.items.put_nowait(None)
            with suppress(BaseException):
                await asyncio.wait_for(reader_task, timeout=0.5)
            if closing is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(closing, timeout=0.5)

    asyncio.run(scenario())
