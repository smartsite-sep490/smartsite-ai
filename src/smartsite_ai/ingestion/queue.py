"""Bounded latest-frame queue enforcing drop-stale backpressure and graceful closure."""

import asyncio
import contextlib

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import IngestionError


class QueueClosedError(IngestionError):
    """Raised when an operation is attempted on a closed BoundedFrameQueue."""


class BoundedFrameQueue:
    """Bounded FIFO queue that drops oldest frames when full to guarantee low-latency processing.

    In live video ingestion, queues must not grow unbounded when downstream consumers
    (e.g., neural network detectors) process slower than camera frame rates.
    This queue drops the stalest frame when capacity is reached and records monotonic
    dropped metrics.

    When closed, pending and future getters are unblocked and raise QueueClosedError.
    """

    def __init__(self, maxsize: int = 5) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        self._maxsize = maxsize
        self._queue: asyncio.Queue[FrameEnvelope] = asyncio.Queue(maxsize=maxsize)
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._dropped_count = 0
        self._closed = False
        self._close_event = asyncio.Event()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def enqueued_count(self) -> int:
        return self._enqueued_count

    @property
    def dequeued_count(self) -> int:
        return self._dequeued_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def close(self) -> None:
        """Mark queue as closed and awaken all pending getters with QueueClosedError."""
        if not self._closed:
            self._closed = True
            self._close_event.set()

    def put(self, frame: FrameEnvelope) -> FrameEnvelope | None:
        """Put a frame envelope into the queue.

        If the queue is full, the oldest stale frame is dropped and returned.

        Raises:
            QueueClosedError: If the queue is already closed.
        """
        if self._closed:
            raise QueueClosedError("Cannot put into a closed queue")

        dropped: FrameEnvelope | None = None
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                self._dropped_count += 1
            except asyncio.QueueEmpty:
                dropped = None

        self._queue.put_nowait(frame)
        self._enqueued_count += 1
        return dropped

    async def get(self) -> FrameEnvelope:
        """Asynchronously retrieve the next available frame envelope.

        Raises:
            QueueClosedError: If queue is closed or closed while waiting.
        """
        # First drain any already available items
        if not self._queue.empty():
            frame = self._queue.get_nowait()
            self._dequeued_count += 1
            return frame

        if self._closed:
            raise QueueClosedError("Queue is closed")

        queue_task = asyncio.create_task(self._queue.get())
        close_task = asyncio.create_task(self._close_event.wait())

        done, pending = await asyncio.wait(
            [queue_task, close_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if queue_task in done:
            frame = queue_task.result()
            self._dequeued_count += 1
            return frame

        # Close event was triggered while waiting
        raise QueueClosedError("Queue is closed")

    def get_nowait(self) -> FrameEnvelope:
        """Retrieve the next frame envelope synchronously without waiting.

        Raises:
            QueueClosedError: If queue is empty and closed.
            asyncio.QueueEmpty: If the queue is empty and not closed.
        """
        if self._queue.empty():
            if self._closed:
                raise QueueClosedError("Queue is closed")
            raise asyncio.QueueEmpty
        frame = self._queue.get_nowait()
        self._dequeued_count += 1
        return frame

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def clear(self) -> None:
        """Drain all current frames from queue without altering monotonic counters."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
