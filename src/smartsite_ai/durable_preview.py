"""Two bounded image purposes observing the same validated ingestion envelope."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from smartsite_ai.configuration_image_observer import ConfigurationImageObserver
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.media_preview import MediaPreviewHub


class DurablePreviewFrames:
    def __init__(
        self, camera: str, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self.raw = MediaPreviewHub(camera, utc_clock=clock)
        self.configuration = ConfigurationImageObserver(camera, clock=clock)

    def offer(self, frame: FrameEnvelope) -> None:
        self.raw.offer(frame)
        self.configuration.offer(frame)

    def clear(self) -> None:
        self.raw.clear()
        self.configuration.clear()

    def latest(self) -> dict[str, Any] | None:
        return self.raw.latest()

    async def close(self) -> None:
        try:
            await self.raw.close()
        finally:
            await self.configuration.close()
