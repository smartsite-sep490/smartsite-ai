"""Candidate producer handoff from a captured observation to the tracked outbox."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import _validate_lowercase_uuid
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences
from smartsite_ai.evidence.models import EvidenceBindingError, build_local_evidence_uri
from smartsite_ai.evidence.protocol import EvidencePublisherProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.pipelines.observation_frame import ObservationFrame
from smartsite_ai.pipelines.ppe_temporal import ConfirmedPpeCandidate, tracked_delivery_frame
from smartsite_ai.pipelines.tracked_frame_encoder import encode_complete_tracked_frame


async def stage_tracked_capture(
    outbox: SqliteEventOutbox,
    frame: ObservationFrame,
    context: EventDeliveryContext,
    credential: WorkerGenerationCredential,
    *,
    site_id: str,
    profile_sha256: str,
    confirmed_ppe_candidates: Iterable[ConfirmedPpeCandidate],
    expires_at: datetime,
    max_pending: int,
    max_bytes: int,
    source_frame: FrameEnvelope | None = None,
    evidence_publisher: EvidencePublisherProtocol | None = None,
    predecessor_wait_seconds: float | None = None,
    poll_interval_seconds: float = 0.25,
    before_enqueue: Callable[[], Awaitable[None]] | None = None,
    on_staged: Callable[[int], None] | None = None,
) -> TrackedFrameReferences:
    """Bind a complete observation to its original context and queue exact parts."""
    if (source_frame is None) != (evidence_publisher is None):
        raise ValueError("Tracked evidence requires both the original frame and a publisher")
    scope_sha256 = compute_canonical_payload_hash(
        {
            **context.model_dump(by_alias=True, exclude={"schema_version", "receipt_command_id"}),
            "siteId": _validate_lowercase_uuid(site_id, field_name="siteId"),
            "profileSha256": profile_sha256,
        }
    )
    predecessor = None
    sequence = frame.source_sequence_number
    if (
        type(sequence) is int
        and sequence > 0
        and any(subject.previous_sequence_number is not None for subject in frame.subject_segments)
    ):
        predecessor = await outbox.read_tracked_predecessor(scope_sha256, str(sequence - 1))
        if predecessor is None and predecessor_wait_seconds is not None:
            try:
                async with asyncio.timeout(predecessor_wait_seconds):
                    while predecessor is None:
                        await asyncio.sleep(poll_interval_seconds)
                        predecessor = await outbox.read_tracked_predecessor(
                            scope_sha256, str(sequence - 1)
                        )
            except TimeoutError:
                raise TimeoutError("Timed out waiting for the tracked predecessor") from None
    parts, references = encode_complete_tracked_frame(
        tracked_delivery_frame(frame, confirmed_ppe_candidates),
        scope_sha256=scope_sha256,
        profile_sha256=profile_sha256,
        predecessor=predecessor,
    )
    if source_frame is not None and evidence_publisher is not None:
        for part in parts:
            if not any(
                observation["type"] == "ZONE_ENTRY"
                or (observation["type"] == "PPE" and observation["status"] == "MISSING")
                for observation in part["observations"]
            ):
                continue
            # The existing publisher validates the exact captured frame and part partition.
            # Use a temporary supported event with the final tracked part ID and copy only evidence.
            publication = TechnicalObservationEvent.model_validate(
                {key: value for key, value in part.items() if key != "continuity"}
                | {"schemaVersion": "1.4.0"}
            )
            published = await evidence_publisher.publish(frame=source_frame, event=publication)
            uri = build_local_evidence_uri(
                session_id=str(source_frame.session_id),
                sequence_number=source_frame.sequence_number,
                event_id=part["eventId"],
            )
            bound = [
                item
                for item in published.evidence
                if item.kind == "FRAME" and item.uri == uri and item.sha256 is not None
            ]
            if len(bound) != 1:
                raise EvidenceBindingError("Tracked evidence must bind to the original part")
            part["evidence"] = [bound[0].model_dump(by_alias=True, exclude_none=True)]

    if before_enqueue is not None:
        await before_enqueue()
    owned = asyncio.create_task(
        outbox.enqueue_tracked_parts(
            credential,
            context,
            parts,
            expected_seal=references,
            expires_at=expires_at,
            max_pending=max_pending,
            max_bytes=max_bytes,
        ),
        name="persist-tracked-frame",
    )
    cancellation = None
    while True:
        try:
            await asyncio.shield(owned)
            break
        except asyncio.CancelledError as error:
            if owned.cancelled():
                raise
            cancellation = error
    if on_staged is not None:
        on_staged(len(parts))
    if cancellation is not None:
        raise cancellation
    return references
