"""Single-reader, one-frame backpressure adapter for LIVE sources only."""

import asyncio
from contextlib import suppress
from typing import Protocol

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import FrameSource, SourceReadError


class MediaSink(Protocol):
    def offer(self, frame: FrameEnvelope) -> None: ...
    def clear(self) -> None: ...


class LiveLatestFrameSource:
    """Drain one capture while inference runs; replace, never queue, older frames.

    Shutdown waits for the in-progress native read before releasing its capture.
    A native decoder that never returns still needs a driver-level read timeout.
    This adapter neither creates another capture nor changes frame timestamps.
    """

    def __init__(self, source: FrameSource) -> None:
        self.source = source
        self._latest: FrameEnvelope | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._stop = False
        self._closed = True
        self._ended = False
        self._failed = False
        self.frames_read = 0
        self.frames_replaced = 0
        self._media_sink: MediaSink | None = None

    def set_media_sink(self, sink: MediaSink) -> None:
        self._media_sink = sink

    @property
    def is_connected(self) -> bool:
        return not self._closed and not self._ended and not self._failed

    async def connect(self) -> None:
        await self.close()
        await self.source.connect()
        self._close_task = None
        self._latest = None
        self._ready = asyncio.Event()
        self._stop = self._closed = self._ended = self._failed = False
        self._task = asyncio.create_task(self._read_loop(), name="live-latest-reader")

    async def _read_loop(self) -> None:
        try:
            while not self._stop:
                frame = await self.source.read_frame()
                if self._stop:
                    break
                if frame is None:
                    self._ended = True
                    break
                self.frames_read += 1
                if self._latest is not None:
                    self.frames_replaced += 1
                self._latest = frame
                if self._media_sink is not None:
                    self._media_sink.offer(frame)
                self._ready.set()
                # Also yield for synthetic/immediately-ready sources.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._failed = True
            raise
        except Exception:
            # Do not retain source error text: it can contain camera credentials.
            self._failed = True
        finally:
            if self._ended or self._failed or self._stop:
                self._latest = None
                if self._media_sink is not None:
                    self._media_sink.clear()
            self._ready.set()

    async def read_frame(self) -> FrameEnvelope | None:
        while True:
            if self._closed or self._failed:
                raise SourceReadError("Live frame reader is unavailable")
            if self._ended:
                return None
            if self._latest is not None:
                frame, self._latest = self._latest, None
                self._ready.clear()
                return frame
            self._ready.clear()
            await self._ready.wait()

    async def close(self) -> None:
        """Finish owned cleanup before propagating cancellation of this caller."""
        self._stop = self._closed = True
        self._latest = None
        self._ready.set()
        if self._media_sink is not None:
            self._media_sink.clear()
        if self._close_task is None:
            task, self._task = self._task, None
            self._close_task = asyncio.create_task(
                self._finish_close(task), name="live-latest-close"
            )
        cancellation_seen = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                # Keep the owned cleanup alive through repeated waiter cancellation.
                cancellation_seen = True
        self._close_task.result()
        if cancellation_seen:
            raise asyncio.CancelledError

    async def _finish_close(self, task: asyncio.Task[None] | None) -> None:
        try:
            if task is not None:
                # A cancelled reader must not cancel an otherwise uncancelled close.
                with suppress(asyncio.CancelledError):
                    await task
        finally:
            await self.source.close()
