"""Domain models and exceptions for observation evidence."""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from smartsite_ai.domain.observations import StrictWireModel

_SAFE_ID_RE = re.compile(r"^[0-9a-zA-Z_-]+$")


class EvidenceError(RuntimeError):
    """Base exception for all evidence operations."""


class EvidenceBindingError(EvidenceError):
    """Raised when a frame cannot be bound to an event."""


class EvidencePathContainmentError(EvidenceError):
    """Raised when evidence path resolution violates containment or traversal checks."""


class EvidenceSizeLimitError(EvidenceError):
    """Raised when an encoded evidence artifact exceeds configured byte limits."""


class EvidenceEncodingError(EvidenceError):
    """Raised when frame encoding fails."""


def build_local_evidence_uri(
    *,
    session_id: UUID | str,
    sequence_number: int,
    event_id: str,
) -> str:
    """Build a provider-neutral local storage URI embedding session, sequence, and eventId."""
    return f"local://evidence/{session_id}/{sequence_number}/{event_id}.jpg"


class EvidenceManifest(StrictWireModel):
    """Deterministic metadata manifest written alongside atomic frame evidence."""

    event_id: str = Field(..., alias="eventId")
    stream_session_id: str = Field(..., alias="streamSessionId")
    sequence_number: int = Field(..., ge=0, alias="sequenceNumber")
    camera_external_id: str = Field(..., alias="cameraExternalId")
    captured_at: str = Field(..., alias="capturedAt")
    uri: str = Field(..., alias="uri")
    media_type: Literal["image/jpeg"] = Field(default="image/jpeg", alias="mediaType")
    sha256: str = Field(..., min_length=64, max_length=64, alias="sha256")
    size_bytes: int = Field(..., ge=1, alias="sizeBytes")
    width: int = Field(..., ge=1, alias="width")
    height: int = Field(..., ge=1, alias="height")

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        stream_session_id: str,
        sequence_number: int,
        camera_external_id: str,
        captured_at: str,
        uri: str,
        sha256: str,
        size_bytes: int,
        width: int,
        height: int,
    ) -> EvidenceManifest:
        return cls.model_validate(
            {
                "eventId": event_id,
                "streamSessionId": stream_session_id,
                "sequenceNumber": sequence_number,
                "cameraExternalId": camera_external_id,
                "capturedAt": captured_at,
                "uri": uri,
                "mediaType": "image/jpeg",
                "sha256": sha256,
                "sizeBytes": size_bytes,
                "width": width,
                "height": height,
            }
        )

    def to_wire_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


__all__ = [
    "EvidenceBindingError",
    "EvidenceEncodingError",
    "EvidenceError",
    "EvidenceManifest",
    "EvidencePathContainmentError",
    "EvidenceSizeLimitError",
    "build_local_evidence_uri",
]
