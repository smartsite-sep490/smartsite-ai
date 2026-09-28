"""Protocol boundary for evidence publishers."""

from __future__ import annotations

from typing import Protocol

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.ingestion.envelope import FrameEnvelope


class EvidencePublisherProtocol(Protocol):
    """Publish frame evidence and return the updated deliverable event."""

    async def publish(
        self,
        *,
        frame: FrameEnvelope,
        event: TechnicalObservationEvent,
    ) -> TechnicalObservationEvent: ...


__all__ = ["EvidencePublisherProtocol"]
