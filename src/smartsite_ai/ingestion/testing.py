"""Deterministic fake frame sources and time/sleep utilities for testing ingestion behavior."""

import asyncio
from datetime import UTC, datetime, timedelta

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError


class FakeFrameSource:
    """Deterministic, controllable frame source simulating camera feeds and failure modes."""

    def __init__(
        self,
        source_id: str,
        initial_frames: list[FrameEnvelope] | None = None,
        connect_failures_before_success: int = 0,
        permanent_connect_failure: bool = False,
        fail_read_once_after: int | None = None,
        secondary_frames: list[FrameEnvelope] | None = None,
        delay_between_frames: float = 0.0,
        is_live: bool = True,
        block_when_exhausted: bool = False,
    ) -> None:
        self._source_id = source_id
        self._frames = list(initial_frames) if initial_frames is not None else []
        self._secondary_frames = list(secondary_frames) if secondary_frames is not None else []
        self._connect_failures_remaining = connect_failures_before_success
        self._permanent_connect_failure = permanent_connect_failure
        self._fail_read_once_after = fail_read_once_after
        self.delay_between_frames = delay_between_frames
        self.is_live = is_live
        self._block_when_exhausted = block_when_exhausted
        self._exhausted_event = asyncio.Event()

        self._is_connected = False
        self._is_closed = False
        self._frames_yielded = 0

        self.connect_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    async def connect(self) -> None:
        self.connect_calls += 1
        if self._is_connected:
            raise SourceConnectionError(
                f"Source {self._source_id} cannot connect while a connection is already open"
            )
        if self._permanent_connect_failure:
            raise SourceConnectionError(
                f"Simulated permanent connection failure for {self._source_id}"
            )

        if self._connect_failures_remaining > 0:
            self._connect_failures_remaining -= 1
            rem = self._connect_failures_remaining
            raise SourceConnectionError(
                f"Simulated transient connection failure ({rem} remaining) for {self._source_id}"
            )

        self._is_connected = True
        self._is_closed = False
        self._exhausted_event.clear()

    async def read_frame(self) -> FrameEnvelope | None:
        if not self._is_connected:
            raise SourceReadError(f"Source {self._source_id} is not connected")

        self.read_calls += 1

        if (
            self._fail_read_once_after is not None
            and self._frames_yielded >= self._fail_read_once_after
        ):
            self._fail_read_once_after = None  # Consume the failure trigger
            self._is_connected = False
            raise SourceReadError(f"Simulated transient read error on source {self._source_id}")

        if self.delay_between_frames > 0.0:
            await asyncio.sleep(self.delay_between_frames)

        if self._frames:
            frame = self._frames.pop(0)
            self._frames_yielded += 1
            return frame

        if self._secondary_frames:
            frame = self._secondary_frames.pop(0)
            self._frames_yielded += 1
            return frame

        if self._block_when_exhausted:
            await self._exhausted_event.wait()

        # Stream exhausted / closed / EOF
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self._is_connected = False
        self._is_closed = True
        self._exhausted_event.set()


class FakeBlockingSource:
    """Frame source that blocks indefinitely on read_frame until explicitly closed.

    Used to verify that cancellation / stop cleanly unblocks hung native / socket reads.
    """

    def __init__(self, source_id: str) -> None:
        self._source_id = source_id
        self._is_connected = False
        self._is_closed = False
        self._unblock_event = asyncio.Event()

        self.connect_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    async def connect(self) -> None:
        self.connect_calls += 1
        if self._is_connected:
            raise SourceConnectionError(
                f"Source {self._source_id} cannot connect while a connection is already open"
            )
        self._is_connected = True
        self._is_closed = False
        self._unblock_event.clear()

    async def read_frame(self) -> FrameEnvelope | None:
        if not self._is_connected:
            raise SourceReadError(f"Source {self._source_id} is not connected")
        self.read_calls += 1
        # Block until close() is called
        await self._unblock_event.wait()
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self._is_connected = False
        self._is_closed = True
        self._unblock_event.set()


class FakeSleeper:
    """Controllable asynchronous sleeper for deterministic backoff testing without real delays."""

    def __init__(self) -> None:
        self.sleep_calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.sleep_calls.append(delay)
        # Yield execution to event loop to allow other coroutines to run
        await asyncio.sleep(0)


class FakeClock:
    """Controllable clock for deterministic timestamping and sampling tests."""

    def __init__(self, start_time: datetime | None = None) -> None:
        self.current_time = start_time or datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += timedelta(seconds=seconds)
