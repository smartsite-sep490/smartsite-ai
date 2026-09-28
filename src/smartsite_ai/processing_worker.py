"""Headless one-camera MF05/MF06 processing worker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evidence.models import FrameBatchBindingError
from smartsite_ai.evidence.protocol import EvidencePublisherProtocol
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.outbox import OutboxCounts, OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines.ppe_temporal import (
    TemporalPpeCandidateGate,
    filter_event_for_delivery,
)

_LOGGER = logging.getLogger(__name__)


def validate_frame_batch_binding(frame: FrameEnvelope, batch: DetectionBatch) -> None:
    """Validate strict exact-frame binding between ingested frame and detection batch."""
    if (
        batch.stream_id != frame.stream_id
        or batch.camera_external_id != frame.camera_external_id
        or batch.session_id != frame.session_id
        or batch.sequence_number != frame.sequence_number
        or batch.captured_at != frame.captured_at
        or batch.frame_width != frame.width
        or batch.frame_height != frame.height
    ):
        raise FrameBatchBindingError(
            f"frame (stream={frame.stream_id}, camera={frame.camera_external_id}, "
            f"session={frame.session_id}, seq={frame.sequence_number}, "
            f"captured_at={frame.captured_at.isoformat()}, dims={frame.width}x{frame.height}) "
            f"does not match detection batch (stream={batch.stream_id}, "
            f"camera={batch.camera_external_id}, session={batch.session_id}, "
            f"seq={batch.sequence_number}, captured_at={batch.captured_at.isoformat()}, "
            f"dims={batch.frame_width}x{batch.frame_height})"
        )


class TechnicalPipelineProtocol(Protocol):
    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent | None: ...


@dataclass(frozen=True, slots=True)
class ProcessingWorkerResult:
    frames_processed: int
    events_enqueued: int
    outbox: OutboxCounts


class ProcessingWorkerSourceError(RuntimeError):
    """The configured source could not sustain a processable stream."""


class HeadlessCameraProcessingWorker:
    """Own one explicit source-to-outbox lifecycle, independent of browser clients."""

    def __init__(
        self,
        *,
        stream: StreamWorker,
        detector: DetectorProtocol,
        pipeline: TechnicalPipelineProtocol,
        ppe_region_id: str,
        configurations: RegionConfigurationStore,
        outbox: SqliteEventOutbox,
        dispatcher: OutboxDispatcher,
        configuration_guard: Callable[[], None] | None = None,
        delivery_interval_seconds: float = 0.25,
        evidence_publisher: EvidencePublisherProtocol | None = None,
    ) -> None:
        if delivery_interval_seconds <= 0:
            raise ValueError("delivery interval must be positive")
        if not ppe_region_id:
            raise ValueError("PPE region ID must not be empty")
        self._stream = stream
        self._detector = detector
        self._pipeline = pipeline
        self._ppe_region_id = ppe_region_id.casefold()
        self._configurations = configurations
        self._outbox = outbox
        self._dispatcher = dispatcher
        self._configuration_guard = configuration_guard
        self._delivery_interval_seconds = delivery_interval_seconds
        self._evidence_publisher = evidence_publisher
        self._stop_delivery = asyncio.Event()
        self._current_session_id: UUID | None = None
        self._current_ppe_geometry_version: int | None = None
        self._temporal_gate = TemporalPpeCandidateGate()
        self._started = False

    async def run(self) -> ProcessingWorkerResult:
        if self._started:
            raise RuntimeError("headless processing worker is single-use")
        self._started = True

        camera_external_id = self._stream.config.camera_external_id
        if await self._configurations.get(camera_external_id) is None:
            raise RuntimeError("camera region configuration must be loaded before worker start")

        await self._outbox.initialize()
        # Replay crash-surviving events before opening a new source session. A
        # transient failure remains durably scheduled and does not lose the event.
        await self._dispatcher.drain_once()
        delivery_task: asyncio.Task[None] | None = None
        frames_processed = 0
        events_enqueued = 0
        frame_task: asyncio.Task[FrameEnvelope] | None = None
        normal_source_completion = False
        try:
            await self._stream.start()
            delivery_task = asyncio.create_task(
                self._delivery_loop(),
                name=f"outbox-dispatch-{self._stream.config.stream_id}",
            )
            while True:
                frame_task = asyncio.create_task(
                    self._stream.get_frame(),
                    name=f"next-frame-{self._stream.config.stream_id}",
                )
                done, _pending = await asyncio.wait(
                    (frame_task, delivery_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if delivery_task in done:
                    if not frame_task.done():
                        frame_task.cancel()
                    await asyncio.gather(frame_task, return_exceptions=True)
                    delivery_task.result()
                    raise RuntimeError("outbox delivery loop stopped unexpectedly")
                try:
                    frame = frame_task.result()
                except QueueClosedError:
                    normal_source_completion = True
                    break
                finally:
                    frame_task = None
                await self._process_frame(frame)
                frames_processed += 1
                if self._last_frame_enqueued:
                    events_enqueued += 1
        finally:
            if frame_task is not None and not frame_task.done():
                frame_task.cancel()
                await asyncio.gather(frame_task, return_exceptions=True)
            self._stop_delivery.set()
            if not normal_source_completion and delivery_task is not None:
                if not delivery_task.done():
                    # Events were persisted before dispatch, so cancelling an
                    # in-flight attempt is safe; restart replays the same IDs.
                    delivery_task.cancel()
                await asyncio.gather(delivery_task, return_exceptions=True)
            await self._stream.stop()
            if normal_source_completion and delivery_task is not None:
                await delivery_task
                await self._dispatcher.drain_once()

        source_status = self._stream.snapshot()
        if source_status.state is StreamState.ERROR:
            raise ProcessingWorkerSourceError("camera stream terminated in an error state")
        if frames_processed == 0:
            raise ProcessingWorkerSourceError("camera stream ended without a processable frame")
        return ProcessingWorkerResult(
            frames_processed=frames_processed,
            events_enqueued=events_enqueued,
            outbox=await self._outbox.counts(),
        )

    async def _delivery_loop(self) -> None:
        while not self._stop_delivery.is_set():
            await self._dispatcher.drain_once()
            try:
                await asyncio.wait_for(
                    self._stop_delivery.wait(),
                    timeout=self._delivery_interval_seconds,
                )
            except TimeoutError:
                continue

    async def _process_frame(self, frame: FrameEnvelope) -> None:
        self._last_frame_enqueued = False
        if self._configuration_guard is not None:
            self._configuration_guard()
        configuration = await self._configurations.get(frame.camera_external_id)
        if configuration is None:
            raise RuntimeError("active camera region configuration became unavailable")
        ppe_region = next(
            (
                region
                for region in configuration.regions
                if region.region_id.casefold() == self._ppe_region_id
            ),
            None,
        )
        if ppe_region is None:
            raise RuntimeError("required PPE region is absent from active camera configuration")

        batch = await self._detector.detect(frame)
        validate_frame_batch_binding(frame, batch)
        if (
            self._current_session_id != batch.session_id
            or self._current_ppe_geometry_version != ppe_region.geometry_version
        ):
            self._current_session_id = batch.session_id
            self._current_ppe_geometry_version = ppe_region.geometry_version
            self._temporal_gate = TemporalPpeCandidateGate()

        event = self._pipeline.process(
            batch,
            region_configuration=configuration,
            event_id=str(
                uuid5(
                    NAMESPACE_URL,
                    "smartsite-headless:"
                    f"{batch.camera_external_id}:{batch.stream_id}:"
                    f"{batch.session_id}:{batch.sequence_number}",
                )
            ),
        )
        active_track_ids: tuple[int, ...] = ()
        ppe_observations = ()
        if event is not None:
            active_track_ids = tuple(
                observation.track_id
                for observation in event.observations
                if observation.type == "PERSON"
            )
            ppe_observations = tuple(
                observation for observation in event.observations if observation.type == "PPE"
            )

        confirmed_candidates = self._temporal_gate.update(
            stream_id=batch.stream_id,
            session_id=batch.session_id,
            observed_at=batch.captured_at,
            active_track_ids=active_track_ids,
            observations=ppe_observations,
        )
        if event is None:
            return
        delivery_event = filter_event_for_delivery(event, confirmed_candidates)
        if delivery_event is not None:
            event_to_enqueue = delivery_event
            if self._evidence_publisher is not None:
                try:
                    event_to_enqueue = await self._evidence_publisher.publish(
                        frame=frame,
                        event=delivery_event,
                    )
                except Exception as error:
                    _LOGGER.warning(
                        "evidence publication failed; continuing durable event delivery",
                        extra={
                            "event_id": delivery_event.event_id,
                            "stream_id": frame.stream_id,
                            "sequence_number": frame.sequence_number,
                            "exception_type": type(error).__name__,
                        },
                    )
                    event_to_enqueue = delivery_event
            await self._outbox.enqueue(event_to_enqueue)
            self._last_frame_enqueued = True


__all__ = [
    "HeadlessCameraProcessingWorker",
    "ProcessingWorkerResult",
    "ProcessingWorkerSourceError",
    "validate_frame_batch_binding",
]
