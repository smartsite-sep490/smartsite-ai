"""One bounded latest-frame buffer; viewer lifetime never owns the producer."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from copy import deepcopy
from typing import Any

from smartsite_ai.media_preview import MediaPreviewHub as MediaPreviewHub


class LatestPreviewHub:
    def __init__(
        self,
        producer: Callable[["LatestPreviewHub"], Awaitable[None]],
        camera_external_id: str,
        resource_lock: asyncio.Lock,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._producer = producer
        self._camera = camera_external_id
        self._lock = resource_lock
        self._clock = clock
        self._frame: dict[str, Any] | None = None
        self._received_at = 0.0
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def started(self) -> bool:
        return self._task is not None

    def latest(self) -> dict[str, Any] | None:
        if self._closed:
            return None
        if self._task is None and not self._lock.locked():
            self._task = asyncio.create_task(self._run())
        if self._frame is None or self._clock() - self._received_at >= 5:
            return None
        return deepcopy(self._frame)

    async def _run(self) -> None:
        try:
            async with self._lock:
                await self._producer(self)
        except asyncio.CancelledError:
            raise
        except Exception:
            # The public endpoint reports unavailable, never producer diagnostics.
            # A terminal task is retained so viewer polling cannot restart inference.
            self._frame = None
        finally:
            self._frame = None

    async def accept(self) -> None:
        """Sink compatibility only; there is no viewer socket to accept."""

    async def send_json(self, data: Any) -> None:
        if (
            not isinstance(data, dict)
            or data.get("type") != "frame"
            or data.get("cameraExternalId") != self._camera
            or len(json.dumps(data, ensure_ascii=True).encode("utf-8")) > 2_000_000
        ):
            self._frame = None
            return
        self._frame = deepcopy(data)
        self._received_at = self._clock()

    async def close(self, code: int = 1000) -> None:
        # stream_realtime closes its sink in finally. Avoid self-cancellation.
        self._closed = True
        self._frame = None
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
