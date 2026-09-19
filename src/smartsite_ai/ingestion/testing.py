"""Deterministic fake frame source for unit testing ingestion behavior."""

import asyncio

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
    ) -> None:
        self._source_id = source_id
        self._frames = list(initial_frames) if initial_frames is not None else []
        self._secondary_frames = list(secondary_frames) if secondary_frames is not None else []
        self._connect_failures_remaining = connect_failures_before_success
        self._permanent_connect_failure = permanent_connect_failure
        self._fail_read_once_after = fail_read_once_after
        self.delay_between_frames = delay_between_frames

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

        # Stream exhausted / closed
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self._is_connected = False
        self._is_closed = True
