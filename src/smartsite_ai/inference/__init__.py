"""Provider-neutral inference domain models and detector boundary."""

from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.inference.protocol import DetectorProtocol

__all__ = [
    "DetectionBatch",
    "DetectorProtocol",
    "NormalizedBoundingBox",
    "NormalizedDetection",
]
