"""Bounded purpose-JPEG observer; never owns or reads a source."""

import asyncio
import copy
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime

from smartsite_ai.configuration_image import (
    ConfigurationImagePayload,
    ConfigurationJpeg,
    configuration_payload,
    encode_configuration_image,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.media_preview import drain_encoder_task


class ConfigurationImageObserver:
    def __init__(
        self,
        camera: str,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        encoder: Callable[[FrameEnvelope], ConfigurationJpeg] = encode_configuration_image,
    ) -> None:
        self.camera_external_id = camera
        self._clock, self._encoder = clock, encoder
        self._pending: FrameEnvelope | None = None
        self._latest: ConfigurationImagePayload | None = None
        self._time: datetime | None = None
        self._generation = 0
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._failed = False
        self.frames_encoded = 0

    def _fresh(self, captured_at: datetime) -> bool:
        return -1 <= (self._clock() - captured_at).total_seconds() <= 2

    def clear(self) -> None:
        self._generation += 1
        self._pending = self._latest = self._time = None

    def offer(self, frame: FrameEnvelope) -> None:
        if (
            self._stop.is_set()
            or self._failed
            or frame.camera_external_id != self.camera_external_id
        ):
            return
        if not self._fresh(frame.captured_at):
            self.clear()
            return
        self._pending = frame
        self._ready.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="configuration-image-observer")

    def latest(self) -> ConfigurationImagePayload | None:
        if self._stop.is_set() or self._failed or self._time is None or not self._fresh(self._time):
            return None
        return copy.deepcopy(self._latest)

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await self._ready.wait()
                self._ready.clear()
                frame, self._pending = self._pending, None
                if frame is None or self._stop.is_set() or not self._fresh(frame.captured_at):
                    continue
                generation = self._generation
                jpeg = await asyncio.to_thread(self._encoder, frame)
                if (
                    not self._stop.is_set()
                    and generation == self._generation
                    and self._fresh(frame.captured_at)
                ):
                    self._latest = configuration_payload(frame, "LIVE", jpeg)
                    self._time = frame.captured_at
                    self.frames_encoded += 1
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), 0.5)
        except Exception:
            self._failed = True
            self.clear()

    async def close(self) -> None:
        self._stop.set()
        self.clear()
        self._ready.set()
        await drain_encoder_task(self._task)
