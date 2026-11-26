"""Headless one-camera MF05/MF06 processing worker."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from smartsite_ai.core.region_configuration_store import CameraBundle, RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.domain.tracked_observations import TrackedScopeTransitionRequired
from smartsite_ai.evidence.models import FrameBatchBindingError
from smartsite_ai.evidence.protocol import EvidencePublisherProtocol
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.outbox import OutboxCounts, OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.integrations.tracked_capture import stage_tracked_capture
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.pipelines.observation_frame import ObservationFrame
from smartsite_ai.pipelines.ppe_temporal import (
    TemporalPpeCandidateGate,
    events_for_delivery,
)
from smartsite_ai.tracking.continuity import FrameContinuityPolicy

_LOGGER = logging.getLogger(__name__)
_LIVE_STAGING_LIMIT = 4


async def _finish_cleanup[T](work: Awaitable[T]) -> T:
    """Drain owned cleanup despite repeated cancellation, then propagate cancellation."""
    cleanup = asyncio.ensure_future(work)
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            if cleanup.cancelled():
                raise
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError
    return result


@dataclass(frozen=True, slots=True)
class _WorkerStateSnapshot:
    restore_pipeline: Callable[[object], None] | None
    pipeline_state: object
    temporal_state: tuple[TemporalPpeCandidateGate, dict[tuple[str, int], TemporalPpeCandidateGate]]
    previous: tuple[object, ...]


@dataclass(slots=True)
class _LiveStagedEntry:
    pre_snapshot: _WorkerStateSnapshot | None
    committed: bool = False


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


@dataclass(frozen=True, slots=True)
class TrackedCapturePolicy:
    credential: WorkerGenerationCredential
    site_id: str
    profile_sha256: str
    pending_seconds: float
    predecessor_wait_seconds: float
    max_pending: int
    max_bytes: int
    continuity_policy: FrameContinuityPolicy | None = None

    def __post_init__(self) -> None:
        if self.continuity_policy is not None and not isinstance(
            self.continuity_policy, FrameContinuityPolicy
        ):
            raise ValueError("Invalid tracked continuity policy")
        if not (
            type(self.pending_seconds) in (int, float)
            and type(self.predecessor_wait_seconds) in (int, float)
            and isfinite(self.pending_seconds)
            and isfinite(self.predecessor_wait_seconds)
            and 0 < self.predecessor_wait_seconds < self.pending_seconds
            and type(self.max_pending) is int
            and self.max_pending > 0
            and type(self.max_bytes) is int
            and self.max_bytes > 0
        ):
            raise ValueError("Invalid tracked capture limits")


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
        partitioned_frame_delivery: bool = False,
        tracked_capture: TrackedCapturePolicy | None = None,
        analysis_observer: Callable[[AnalysisFrame], None] | None = None,
        experimental_object_preview: bool = False,
        delivery_context_provider: Callable[
            [CameraBundle, FrameEnvelope, Literal["LIVE", "REPLAY"]],
            Awaitable[EventDeliveryContext | None],
        ]
        | None = None,
    ) -> None:
        if delivery_interval_seconds <= 0:
            raise ValueError("delivery interval must be positive")
        if not ppe_region_id:
            raise ValueError("PPE region ID must not be empty")
        if tracked_capture is not None and (
            partitioned_frame_delivery or delivery_context_provider is None
        ):
            raise ValueError("Tracked capture requires bound context and its own delivery route")
        if tracked_capture is not None and not stream.config.is_live:
            raise ValueError("Tracked capture requires the Backend LIVE admission route")
        self._tracked_capture = tracked_capture
        self._stream = stream
        self._detector = detector
        self._analysis_observer = analysis_observer
        self._experimental_object_preview = experimental_object_preview
        self._pipeline = pipeline
        self._ppe_region_id = ppe_region_id.casefold()
        self._configurations = configurations
        self._outbox = outbox
        self._dispatcher = dispatcher
        self._configuration_guard = configuration_guard
        self._delivery_interval_seconds = delivery_interval_seconds
        self._evidence_publisher = evidence_publisher
        self._partitioned_frame_delivery = partitioned_frame_delivery
        self._delivery_context_provider = delivery_context_provider
        self._last_frame_event_count = 0
        self._live_staging_queue: asyncio.Queue[Callable[[], Awaitable[None]] | None] | None = None
        self._live_staged_event_count = 0
        self._live_staged_entries: deque[_LiveStagedEntry] = deque()
        self._current_frame_snapshot: _WorkerStateSnapshot | None = None
        self._stop_delivery = asyncio.Event()
        self._current_session_id: UUID | None = None
        self._current_ppe_geometry_version: int | None = None
        self._temporal_gate = self._new_temporal_gate()
        self._region_gates: dict[tuple[str, int], TemporalPpeCandidateGate] = {}
        self._last_tracked_sequence: int | None = None
        self._last_tracked_batch: DetectionBatch | None = None
        self._active_snapshot: CameraRegionConfiguration | CameraBundle | None = None
        self._started = False

    async def _stage_live_frames(
        self, queue: asyncio.Queue[Callable[[], Awaitable[None]] | None]
    ) -> None:
        while (stage := await queue.get()) is not None:
            await stage()

    async def _capture_frames(self) -> tuple[int, int]:
        frames_processed = 0
        events_enqueued = 0
        while True:
            try:
                frame = await self._stream.get_frame()
            except QueueClosedError:
                if self._live_staging_queue is not None:
                    await self._live_staging_queue.put(None)
                return frames_processed, events_enqueued
            await self._process_frame(frame)
            frames_processed += 1
            if self._last_frame_enqueued:
                events_enqueued += self._last_frame_event_count

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
        staging_task: asyncio.Task[None] | None = None
        capture_task: asyncio.Task[tuple[int, int]] | None = None
        normal_source_completion = False
        try:
            await self._stream.start()
            delivery_task = asyncio.create_task(
                self._delivery_loop(),
                name=f"outbox-dispatch-{self._stream.config.stream_id}",
            )
            if self._stream.config.is_live and self._tracked_capture is not None:
                self._live_staging_queue = asyncio.Queue(
                    maxsize=min(_LIVE_STAGING_LIMIT, self._tracked_capture.max_pending)
                )
                staging_task = asyncio.create_task(
                    self._stage_live_frames(self._live_staging_queue), name="stage-live-frames"
                )
            capture_task = asyncio.create_task(self._capture_frames(), name="process-camera-frames")
            workers: set[asyncio.Task[tuple[int, int]] | asyncio.Task[None]] = {capture_task}
            if staging_task is not None:
                workers.add(staging_task)
            while workers:
                done, _pending = await asyncio.wait(
                    (*workers, delivery_task), return_when=asyncio.FIRST_COMPLETED
                )
                if delivery_task in done:
                    delivery_task.result()
                    raise RuntimeError("outbox delivery loop stopped unexpectedly")
                for worker in done:
                    worker.result()
                workers.difference_update(done)
                if staging_task is not None and staging_task in done and not capture_task.done():
                    raise RuntimeError("live staging loop stopped unexpectedly")
            frames_processed, events_enqueued = capture_task.result()
            events_enqueued += self._live_staged_event_count
            normal_source_completion = True
        finally:
            workers_to_stop = [task for task in (capture_task, staging_task) if task is not None]
            for task in workers_to_stop:
                if not task.done():
                    task.cancel()
            try:
                await _finish_cleanup(asyncio.gather(*workers_to_stop, return_exceptions=True))
            finally:
                self._restore_first_uncommitted_live_state()
                self._live_staging_queue = None
                self._stop_delivery.set()
                try:
                    if not normal_source_completion and delivery_task is not None:
                        if not delivery_task.done():
                            # Events were persisted before dispatch, so cancelling an
                            # in-flight attempt is safe; restart replays the same IDs.
                            delivery_task.cancel()
                        await _finish_cleanup(asyncio.gather(delivery_task, return_exceptions=True))
                finally:
                    await _finish_cleanup(self._stream.stop())
            if normal_source_completion and delivery_task is not None:
                await delivery_task
                await self._dispatcher.drain_once()
                tracked = self._tracked_capture
                if tracked is not None:
                    try:
                        while (
                            remaining := await self._outbox.tracked_pending_remaining_seconds(
                                tracked.credential
                            )
                        ) is not None:
                            if remaining <= 0:
                                raise TimeoutError
                            async with asyncio.timeout(remaining):
                                await self._dispatcher.drain_once()
                                if await self._outbox.has_pending_tracked_frame(tracked.credential):
                                    await asyncio.sleep(self._delivery_interval_seconds)
                    except TimeoutError:
                        raise TimeoutError(
                            "Timed out waiting for final tracked frame confirmation"
                        ) from None

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

    def _capture_state_snapshot(self) -> _WorkerStateSnapshot:
        checkpoint = getattr(self._pipeline, "checkpoint_derived_state", None)
        restore = getattr(self._pipeline, "restore_derived_state", None)
        if checkpoint is None or restore is None:
            checkpoint = getattr(self._pipeline, "checkpoint_temporal_state", None)
            restore = getattr(self._pipeline, "restore_temporal_state", None)
        pipeline_state = checkpoint() if checkpoint is not None and restore is not None else None
        return _WorkerStateSnapshot(
            restore_pipeline=restore if checkpoint is not None and restore is not None else None,
            pipeline_state=pipeline_state,
            temporal_state=deepcopy((self._temporal_gate, self._region_gates)),
            previous=(
                self._current_session_id,
                self._current_ppe_geometry_version,
                self._active_snapshot,
                self._last_tracked_sequence,
                self._last_tracked_batch,
            ),
        )

    def _restore_state_snapshot(self, snapshot: _WorkerStateSnapshot) -> None:
        if snapshot.restore_pipeline is not None:
            snapshot.restore_pipeline(snapshot.pipeline_state)
        self._temporal_gate, self._region_gates = snapshot.temporal_state
        (
            session_id,
            geometry_version,
            active_snapshot,
            last_sequence,
            last_batch,
        ) = snapshot.previous
        self._current_session_id = session_id  # type: ignore[assignment]
        self._current_ppe_geometry_version = geometry_version  # type: ignore[assignment]
        self._active_snapshot = active_snapshot  # type: ignore[assignment]
        self._last_tracked_sequence = last_sequence  # type: ignore[assignment]
        self._last_tracked_batch = last_batch  # type: ignore[assignment]

    def _restore_first_uncommitted_live_state(self) -> None:
        for entry in self._live_staged_entries:
            if not entry.committed and entry.pre_snapshot is not None:
                self._restore_state_snapshot(entry.pre_snapshot)
                break
        self._live_staged_entries.clear()

    async def _process_frame(self, frame: FrameEnvelope) -> None:
        snapshot = self._capture_state_snapshot()
        self._current_frame_snapshot = snapshot
        self._discard_frame = False
        self._durably_committed = False
        completed = False
        try:
            await self._process_frame_once(frame)
            completed = not self._discard_frame
        finally:
            self._current_frame_snapshot = None
            if not completed and not self._durably_committed:
                self._restore_state_snapshot(snapshot)

    def _new_temporal_gate(self) -> TemporalPpeCandidateGate:
        policy = (
            self._tracked_capture.continuity_policy if self._tracked_capture is not None else None
        )
        expiry = (
            max(timedelta(seconds=1), policy.max_capture_gap)
            if policy is not None and policy.max_capture_gap is not None
            else timedelta(seconds=1)
        )
        return TemporalPpeCandidateGate(track_expiry=expiry)

    async def _process_frame_once(self, frame: FrameEnvelope) -> None:
        self._last_frame_enqueued = False
        self._last_frame_event_count = 0
        if self._configuration_guard is not None:
            self._configuration_guard()
        configuration = await self._configurations.get_active_configuration(
            frame.camera_external_id
        )
        if configuration is None:
            raise RuntimeError("active camera region configuration became unavailable")
        if (
            self._tracked_capture is not None
            and self._active_snapshot is not None
            and (
                not isinstance(
                    configuration, (CameraConfigurationBundle, CameraPolicyConfigurationBundle)
                )
                or not isinstance(
                    self._active_snapshot,
                    (CameraConfigurationBundle, CameraPolicyConfigurationBundle),
                )
                or (
                    configuration.configuration_version,
                    configuration.bundle_sha256,
                    configuration.view_revision,
                )
                != (
                    self._active_snapshot.configuration_version,
                    self._active_snapshot.bundle_sha256,
                    self._active_snapshot.view_revision,
                )
            )
        ):
            raise TrackedScopeTransitionRequired(
                "Tracked source requires a trusted scope transition"
            )
        delivery_context = None
        if self._delivery_context_provider is not None:
            if not isinstance(
                configuration, (CameraConfigurationBundle, CameraPolicyConfigurationBundle)
            ):
                raise RuntimeError("Bound delivery requires an immutable configuration bundle")
            delivery_context = await self._delivery_context_provider(
                configuration, frame, "LIVE" if self._stream.config.is_live else "REPLAY"
            )
            if delivery_context is None:
                if not self._stream.config.is_live:
                    raise RuntimeError("Replay frame has no confirmed delivery context")
                self._discard_frame = True
                return
        ppe_region = next(
            (
                region
                for region in configuration.regions
                if region.region_id.casefold() == self._ppe_region_id
            ),
            None,
        )
        if ppe_region is None and not isinstance(
            configuration, (CameraConfigurationBundle, CameraPolicyConfigurationBundle)
        ):
            raise RuntimeError("required PPE region is absent from active camera configuration")

        batch = await self._detector.detect(frame)
        validate_frame_batch_binding(frame, batch)
        if self._configuration_guard is not None:
            self._configuration_guard()
        if (
            self._stream.config.is_live
            and await self._configurations.get_active_configuration(frame.camera_external_id)
            is not configuration
        ):
            if self._tracked_capture is not None:
                raise TrackedScopeTransitionRequired(
                    "Tracked source requires a trusted scope transition"
                )
            return
        if (
            self._current_session_id != batch.session_id
            or self._active_snapshot is not configuration
        ):
            self._current_session_id = batch.session_id
            self._active_snapshot = configuration
            self._current_ppe_geometry_version = ppe_region.geometry_version if ppe_region else None
            self._temporal_gate = self._new_temporal_gate()
            self._last_tracked_sequence = None
            self._last_tracked_batch = None
            self._region_gates = {}

        observe = getattr(
            self._pipeline,
            "observe",
            self._pipeline.process if hasattr(self._pipeline, "process") else None,
        )
        if self._tracked_capture is not None:
            observe = getattr(self._pipeline, "observe_frame", None)
        if observe is None:
            raise TypeError("Pipeline must provide observe or process")
        event = observe(
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

        if isinstance(configuration, CameraPolicyConfigurationBundle):
            if (
                self._tracked_capture is None
                or not isinstance(event, ObservationFrame)
                or not event.requires_tracked_wire
            ):
                raise ProcessingWorkerSourceError("PPE policy v2 requires tracked v1.7 capture")
            if self._last_tracked_sequence is not None and (
                not self._tracked_capture.continuity_policy.allows(self._last_tracked_batch, batch)
                if self._tracked_capture.continuity_policy is not None
                and self._last_tracked_batch is not None
                else batch.sequence_number != self._last_tracked_sequence + 1
            ):
                self._temporal_gate = self._new_temporal_gate()
            self._last_tracked_sequence = batch.sequence_number
            self._last_tracked_batch = batch
            confirmed_candidates = self._temporal_gate.update(
                stream_id=batch.stream_id,
                session_id=batch.session_id,
                observed_at=batch.captured_at,
                active_track_ids=active_track_ids,
                observations=ppe_observations,
                camera_configuration_version=configuration.configuration_version,
            )
        elif isinstance(configuration, CameraConfigurationBundle):
            confirmed_candidates = []
            for region in configuration.regions:
                key = (region.region_id, region.geometry_version)
                gate = self._region_gates.setdefault(key, TemporalPpeCandidateGate())
                confirmed_candidates.extend(
                    replace(
                        candidate,
                        region_id=region.region_id,
                        geometry_version=region.geometry_version,
                    )
                    for candidate in gate.update(
                        stream_id=batch.stream_id,
                        session_id=batch.session_id,
                        observed_at=batch.captured_at,
                        active_track_ids=active_track_ids,
                        observations=tuple(
                            obs
                            for obs in ppe_observations
                            if obs.region_id == region.region_id
                            and obs.geometry_version == region.geometry_version
                        ),
                    )
                )
        else:
            confirmed_candidates = self._temporal_gate.update(
                stream_id=batch.stream_id,
                session_id=batch.session_id,
                observed_at=batch.captured_at,
                active_track_ids=active_track_ids,
                observations=ppe_observations,
            )
        if self._analysis_observer is not None:
            try:
                # Preview shows retained confirmation, not only new delivery candidates.
                preview_gates = (
                    tuple(
                        self._region_gates[(region.region_id, region.geometry_version)]
                        for region in configuration.regions
                    )
                    if isinstance(configuration, CameraConfigurationBundle)
                    else (self._temporal_gate,)
                )
                confirmed_preview_items = frozenset(
                    item
                    for gate in preview_gates
                    for item in gate.confirmed_track_items(
                        stream_id=batch.stream_id,
                        session_id=batch.session_id,
                        active_track_ids=active_track_ids,
                    )
                )
                occupancy = getattr(self._pipeline, "occupied_zone_track_regions", None)
                occupied_zone_regions = (
                    occupancy(
                        stream_id=batch.stream_id,
                        session_id=batch.session_id,
                        active_track_ids=active_track_ids,
                    )
                    if occupancy is not None
                    else frozenset()
                )
                self._analysis_observer(
                    AnalysisFrame(
                        frame=frame,
                        observation=event,
                        configuration=configuration,
                        ppe_region_id=self._ppe_region_id,
                        confirmed_ppe_items=confirmed_preview_items,
                        occupied_zone_regions=frozenset(occupied_zone_regions),
                        zone_anchor=getattr(self._pipeline, "zone_anchor", "BOTTOM_CENTER"),
                        raw_detections=batch if self._experimental_object_preview else None,
                    )
                )
            except Exception:
                # Preview cannot disable durable event delivery. A failed sink
                # is retired for this worker; only a safe diagnostic is logged.
                self._analysis_observer = None
                _LOGGER.warning("analysis preview observer failed; retired until runtime restart")
        tracked = self._tracked_capture
        if tracked is not None:
            if (
                not isinstance(event, ObservationFrame)
                or not isinstance(
                    configuration, (CameraConfigurationBundle, CameraPolicyConfigurationBundle)
                )
                or event.requires_tracked_wire
                != isinstance(configuration, CameraPolicyConfigurationBundle)
                or delivery_context is None
                or delivery_context.camera_id != configuration.camera_id
                or delivery_context.configuration_version != configuration.configuration_version
                or delivery_context.view_revision != configuration.view_revision
                or delivery_context.bundle_sha256 != configuration.bundle_sha256
                or delivery_context.source_kind
                != ("LIVE" if self._stream.config.is_live else "REPLAY")
            ):
                raise ProcessingWorkerSourceError(
                    "Tracked capture context is unavailable or changed"
                )

            camera_external_id = frame.camera_external_id
            staging_queue = self._live_staging_queue
            staged_entry = (
                _LiveStagedEntry(pre_snapshot=self._current_frame_snapshot)
                if staging_queue is not None
                else None
            )
            # Retain pixels only for frames that can produce tracked evidence.
            evidence_publisher = self._evidence_publisher
            if not confirmed_candidates and not any(
                observation.type == "ZONE_ENTRY" for observation in event.observations
            ):
                evidence_publisher = None
            evidence_frame = frame if evidence_publisher is not None else None

            async def check_context() -> None:
                if self._configuration_guard is not None:
                    self._configuration_guard()
                if (
                    self._stream.config.is_live
                    and await self._configurations.get_active_configuration(camera_external_id)
                    is not configuration
                ):
                    raise TrackedScopeTransitionRequired(
                        "Tracked source requires a trusted scope transition"
                    )

            def on_staged(parts: int) -> None:
                if staging_queue is not None:
                    if staged_entry is not None:
                        staged_entry.committed = True
                        staged_entry.pre_snapshot = None
                        while self._live_staged_entries and self._live_staged_entries[0].committed:
                            self._live_staged_entries.popleft()
                    self._live_staged_event_count += parts
                    return
                self._durably_committed = True
                self._last_frame_enqueued = True
                self._last_frame_event_count = parts

            async def stage() -> None:
                if staging_queue is not None:
                    # The queued observation may have waited across a configuration change.
                    await check_context()
                now = datetime.now(UTC)
                credential_expiry = datetime.fromisoformat(tracked.credential.expires_at)
                tracked.credential.ensure_current(now)
                expires_at = min(
                    now + timedelta(seconds=tracked.pending_seconds),
                    credential_expiry,
                )
                await stage_tracked_capture(
                    self._outbox,
                    event,
                    delivery_context,
                    tracked.credential,
                    site_id=tracked.site_id,
                    profile_sha256=tracked.profile_sha256,
                    confirmed_ppe_candidates=confirmed_candidates,
                    expires_at=expires_at,
                    max_pending=tracked.max_pending,
                    max_bytes=tracked.max_bytes,
                    source_frame=evidence_frame,
                    evidence_publisher=evidence_publisher,
                    predecessor_wait_seconds=tracked.predecessor_wait_seconds,
                    poll_interval_seconds=self._delivery_interval_seconds,
                    before_enqueue=check_context,
                    on_staged=on_staged,
                    continuity_policy=tracked.continuity_policy,
                )

            if staging_queue is not None:
                assert staged_entry is not None
                self._live_staged_entries.append(staged_entry)
                try:
                    await staging_queue.put(stage)
                except BaseException:
                    self._live_staged_entries.remove(staged_entry)
                    raise
            else:
                await stage()
            return

        if event is None:
            return
        delivery_events = events_for_delivery(
            event,
            confirmed_candidates,
            partition_sequence_number=frame.sequence_number
            if self._partitioned_frame_delivery
            else None,
        )
        prepared_events = []
        for delivery_event in delivery_events:
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
            prepared_events.append(event_to_enqueue)
        if not prepared_events:
            return
        if self._configuration_guard is not None:
            self._configuration_guard()
        if (
            self._stream.config.is_live
            and await self._configurations.get_active_configuration(frame.camera_external_id)
            is not configuration
        ):
            self._discard_frame = True
            return

        async def persist():
            if delivery_context is not None:
                await self._outbox.enqueue_many(
                    tuple(prepared_events), delivery_context=delivery_context
                )
                return
            if len(prepared_events) == 1:
                await self._outbox.enqueue(prepared_events[0])
            else:
                await self._outbox.enqueue_many(tuple(prepared_events))

        owned = asyncio.create_task(persist(), name="persist-observation-frame")
        cancellation = None
        while True:
            try:
                await asyncio.shield(owned)
                break
            except asyncio.CancelledError as error:
                if owned.cancelled():
                    raise
                cancellation = error
        self._durably_committed = True
        self._last_frame_enqueued = True
        self._last_frame_event_count = len(prepared_events)
        if cancellation is not None:
            raise cancellation


__all__ = [
    "HeadlessCameraProcessingWorker",
    "ProcessingWorkerResult",
    "ProcessingWorkerSourceError",
    "TrackedCapturePolicy",
    "validate_frame_batch_binding",
]
