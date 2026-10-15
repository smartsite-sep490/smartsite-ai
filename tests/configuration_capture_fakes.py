"""Synthetic source fixtures: no camera adapter, model, network or filesystem I/O."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from smartsite_ai.ingestion.envelope import FrameEnvelope

NOW = datetime(2026, 10, 15, 14, 30, tzinfo=UTC)
SESSION = UUID("66d47480-660f-4e12-951e-fd67949f5213")


class FakeSource:
    source_id = "configuration-stream"

    def __init__(self) -> None:
        self.items: asyncio.Queue[FrameEnvelope | Exception | None] = asyncio.Queue()
        self.connected = asyncio.Event()
        self.reading = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.close_gate.set()
        self.connect_gate = asyncio.Event()
        self.connect_gate.set()
        self.connects = 0
        self.closes = 0
        self.reads = 0
        self.is_connected = False
        self.close_error = False

    async def connect(self) -> None:
        self.connects += 1
        self.connected.set()
        await self.connect_gate.wait()
        self.is_connected = True

    async def read_frame(self) -> FrameEnvelope | None:
        self.reads += 1
        self.reading.set()
        value = await self.items.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closes += 1
        await self.close_gate.wait()
        if self.close_error:
            raise RuntimeError("rtsp://private-user:private-secret@camera")
        self.is_connected = False


def frame(sequence: int = 0, **changes: object) -> FrameEnvelope:
    return FrameEnvelope.model_validate(
        {
            "stream_id": "configuration-stream",
            "camera_external_id": "camera-a",
            "session_id": SESSION,
            "sequence_number": sequence,
            "captured_at": NOW,
            "width": 64,
            "height": 64,
            "payload": b"\x00\x20\x40" * 64 * 64,
            **changes,
        }
    )


async def until(predicate) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(capture, source: FakeSource) -> None:
    source.connect_gate.set()
    source.close_gate.set()
    source.items.put_nowait(None)
    assert await capture.stop()
