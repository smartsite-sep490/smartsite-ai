"""Domain models and exceptions for observation evidence."""

from __future__ import annotations

from uuid import UUID


class EvidenceError(RuntimeError):
    """Base exception for all evidence operations."""


class EvidenceBindingError(EvidenceError):
    """Raised when a frame cannot be bound to an event."""


class FrameBatchBindingError(EvidenceBindingError):
    """Raised when FrameEnvelope and DetectionBatch attributes do not match."""


class EvidencePathContainmentError(EvidenceError):
    """Raised when evidence path resolution violates containment or traversal checks."""


class EvidenceSizeLimitError(EvidenceError):
    """Raised when an encoded evidence artifact exceeds configured byte limits."""


class EvidenceEncodingError(EvidenceError):
    """Raised when frame encoding fails."""


class EvidenceConflictError(EvidenceError):
    """Raised when an evidence artifact already exists with conflicting payload content."""


def build_local_evidence_uri(
    *,
    session_id: UUID | str,
    sequence_number: int,
    event_id: str,
) -> str:
    """Build a provider-neutral local storage URI embedding session, sequence, and eventId."""
    return f"local://evidence/{session_id}/{sequence_number}/{event_id}.jpg"


__all__ = [
    "EvidenceBindingError",
    "EvidenceConflictError",
    "EvidenceEncodingError",
    "EvidenceError",
    "EvidencePathContainmentError",
    "EvidenceSizeLimitError",
    "FrameBatchBindingError",
    "build_local_evidence_uri",
]
