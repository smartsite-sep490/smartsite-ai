"""Share one detector across cameras without assuming provider thread safety."""

from __future__ import annotations

import asyncio

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope


def runner_requires_serialization(runner: object) -> bool:
    """Serialize unless the loaded runner explicitly proves concurrent access safe."""

    return getattr(runner, "concurrent_inference_safe", False) is not True


class SerializingDetector:
    """Hold one detector call at a time, including work dispatched to a thread."""

    def __init__(self, detector: DetectorProtocol) -> None:
        self._detector = detector
        self._lock = asyncio.Lock()

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        async with self._lock:
            return await self._detector.detect(frame)


def share_detector(detector: DetectorProtocol, runner: object) -> DetectorProtocol:
    """Return the detector, wrapping it when the runner is not proven thread-safe."""

    if runner_requires_serialization(runner):
        return SerializingDetector(detector)
    return detector


__all__ = [
    "SerializingDetector",
    "runner_requires_serialization",
    "share_detector",
]
