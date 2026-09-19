"""Bounded latest-frame queue enforcing drop-stale backpressure policy."""

import asyncio

from smartsite_ai.ingestion.envelope import FrameEnvelope


class BoundedFrameQueue:
    """Bounded FIFO queue that drops oldest frames when full to guarantee low-latency processing.

    In live video ingestion, queues must not grow unbounded when downstream consumers
    (e.g., neural network detectors) process slower than camera frame rates.
    This queue drops the stalest frame when capacity is reached and records monotonic
    dropped metrics.
    """

    def __init__(self, maxsize: int = 5) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        self._maxsize = maxsize
        self._queue: asyncio.Queue[FrameEnvelope] = asyncio.Queue(maxsize=maxsize)
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._dropped_count = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def enqueued_count(self) -> int:
        return self._enqueued_count

    @property
    def dequeued_count(self) -> int:
        return self._dequeued_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def put(self, frame: FrameEnvelope) -> FrameEnvelope | None:
        """Put a frame envelope into the queue.

        If the queue is full, the oldest stale frame is dropped and returned.
        """
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
        """Asynchronously retrieve the next available frame envelope."""
        frame = await self._queue.get()
        self._dequeued_count += 1
        return frame

    def get_nowait(self) -> FrameEnvelope:
        """Retrieve the next frame envelope synchronously without waiting.

        Raises:
            asyncio.QueueEmpty: If the queue is empty.
        """
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
