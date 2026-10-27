"""Bounded raw display fanout, independent of detector cadence and evidence."""

import asyncio
import base64
import io
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from smartsite_ai.ingestion.envelope import FrameEnvelope


def encode_media_frame(frame: FrameEnvelope) -> tuple[bytes, int, int]:
    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
    image.thumbnail((640, 640))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=65)
    encoded = output.getvalue()
    if len(encoded) > 131072:
        raise ValueError("Raw media frame exceeds display budget")
    return encoded, image.width, image.height


class MediaPreviewHub:
    """One pending frame/one encoder for all readers, never a historical queue."""

    def __init__(
        self,
        camera: str,
        *,
        encoder: Callable[[FrameEnvelope], bytes | tuple[bytes, int, int]] = encode_media_frame,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self._encoder = encoder
        self._clock = clock
        self._pending: FrameEnvelope | None = None
        self._frame: dict[str, Any] | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._generation = 0
        self._received_at = 0.0
        self._closed = False
        self.frames_encoded = 0

    def clear(self) -> None:
        self._generation += 1
        self._pending = None
        self._frame = None

    def offer(self, frame: FrameEnvelope) -> None:
        if self._closed or frame.camera_external_id != self.camera:
            return
        self._pending = frame
        self._ready.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="raw-media-encoder")

    async def _run(self) -> None:
        last_encode = -1.0
        while not self._closed:
            await self._ready.wait()
            delay = max(0.0, 0.125 - (self._clock() - last_encode))
            if delay:
                await asyncio.sleep(delay)
            self._ready.clear()
            frame, self._pending = self._pending, None
            if frame is None:
                continue
            generation = self._generation
            try:
                result = await asyncio.to_thread(self._encoder, frame)
                encoded, width, height = (
                    result if isinstance(result, tuple) else (result, frame.width, frame.height)
                )
                last_encode = self._clock()
                if self._closed or generation != self._generation:
                    continue
                if not encoded.startswith(b"\xff\xd8") or len(encoded) > 131072:
                    self._frame = None
                    continue
                self._frame = {
                    "type": "media-frame",
                    "mediaVersion": 1,
                    "cameraExternalId": frame.camera_external_id,
                    "sessionId": str(frame.session_id),
                    "sequenceNumber": str(frame.sequence_number),
                    "capturedAt": frame.captured_at_iso,
                    "width": width,
                    "height": height,
                    "imageDataUrl": "data:image/jpeg;base64,"
                    + base64.b64encode(encoded).decode("ascii"),
                }
                self._received_at = self._clock()
                self.frames_encoded += 1
            except Exception:
                self._frame = None

    def latest(self) -> dict[str, Any] | None:
        if self._closed or self._frame is None or self._clock() - self._received_at >= 2:
            return None
        age = (
            datetime.now(UTC) - datetime.fromisoformat(self._frame["capturedAt"])
        ).total_seconds()
        if not -1 <= age < 2:
            return None
        return dict(self._frame)

    async def close(self) -> None:
        self._closed = True
        self.clear()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
