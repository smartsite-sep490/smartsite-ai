"""Provider-neutral inference domain models and detector boundary."""

from smartsite_ai.inference.identity import (
    FaceEnrollmentRequest,
    FaceEnrollmentResult,
    FaceEnrollmentSample,
    FaceRecognizerProtocol,
    FaceVerificationFrame,
    FaceVerificationResult,
    UnavailableFaceRecognizer,
)
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.inference.protocol import DetectorProtocol

__all__ = [
    "DetectionBatch",
    "DetectorProtocol",
    "FaceRecognizerProtocol",
    "FaceEnrollmentRequest",
    "FaceEnrollmentResult",
    "FaceEnrollmentSample",
    "FaceVerificationFrame",
    "FaceVerificationResult",
    "NormalizedBoundingBox",
    "NormalizedDetection",
    "UnavailableFaceRecognizer",
]
