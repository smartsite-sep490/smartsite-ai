"""Build per-camera MF05/MF06 sessions that share one detector."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.inference.ppe_profiles import ExperimentalPpeProfile, experimental_ppe_profile
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.frame_evidence_upload import TrackedFrameEvidenceUploader
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.integrations.worker_generation import (
    GenerationBoundSource,
    WorkerGenerationCredential,
    load_worker_generation,
)
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, TrackedCapturePolicy
from smartsite_ai.runtime.analysis_preview import AnalysisPreviewObserver
from smartsite_ai.runtime.manifest import BoundCamera, BoundManifest, CameraRuntimeError
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.supervisor import (
    CameraOutcome,
    CameraSession,
    _finish_cleanup,
    startup_outcomes,
)
from smartsite_ai.tracking.iou_tracker import IoUPersonTracker

SourceFactory = Callable[[StreamConfig], object]
BackendClientFactory = Callable[[], BackendClient]
ConfigurationClientFactory = Callable[[], BackendConfigurationClient]


@dataclass(frozen=True, slots=True)
class CameraPlan:
    bound: BoundCamera
    store: RegionConfigurationStore
    poller: CameraConfigurationPoller
    configuration_client: BackendConfigurationClient
    generation: WorkerGenerationCredential | None = None


async def preflight_manifest(
    bound: BoundManifest,
    *,
    configuration_client_factory: ConfigurationClientFactory,
    policy_v2_client_factory: ConfigurationClientFactory | None = None,
) -> tuple[tuple[CameraOutcome, ...] | None, tuple[CameraPlan, ...]]:
    """Confirm every camera region before a model or source is opened."""

    plans: list[CameraPlan] = []
    identities = [(camera.stream_id, camera.camera_external_id) for camera in bound.cameras]
    try:
        for camera in bound.cameras:
            tracked = camera.entry.tracked_v17 is not None
            factory = policy_v2_client_factory if tracked else configuration_client_factory
            if factory is None:
                raise CameraRuntimeError("Tracked v1.7 requires an explicit v2 client")
            client = factory()
            try:
                generation = None
                if camera.entry.generation_credential_file is not None:
                    generation = load_worker_generation(
                        Path(camera.entry.generation_credential_file),
                        camera.entry.camera_id,
                        now=datetime.now(UTC),
                    )
                    await SqliteEventOutbox(camera.outbox).claim_worker_session(
                        camera.entry.camera_id,
                        UUID(generation.worker_generation_id),
                        UUID(generation.stream_session_id),
                    )
                store = RegionConfigurationStore(allow_policy_v2=tracked)
                poller = CameraConfigurationPoller(
                    client=client,
                    store=store,
                    camera_id=camera.entry.camera_id,
                    camera_external_id=camera.camera_external_id,
                    poll_interval_seconds=bound.manifest.poll_interval_seconds,
                    stale_after_seconds=bound.manifest.stale_after_seconds,
                    require_bundle=generation is not None
                    or camera.entry.partitioned_frame_delivery
                    or tracked,
                    pin_bundle_scope=tracked,
                )
                await poller.refresh()
                poller.ensure_fresh()
                if tracked and not isinstance(
                    await store.get_bundle(camera.camera_external_id),
                    CameraPolicyConfigurationBundle,
                ):
                    raise CameraRuntimeError("Tracked v1.7 requires a v2 configuration bundle")
                await _require_ppe_region(
                    store, camera.camera_external_id, camera.entry.ppe_region_id
                )
            except BaseException:
                await _finish_cleanup(client.aclose())
                raise
            plans.append(
                CameraPlan(
                    bound=camera,
                    store=store,
                    poller=poller,
                    configuration_client=client,
                    generation=generation,
                )
            )
    except asyncio.CancelledError:
        await _finish_cleanup(_close_plans(plans))
        raise
    except Exception as exc:
        await _finish_cleanup(_close_plans(plans))
        error_class = type(exc).__name__
        failed_stream = bound.cameras[len(plans)].stream_id
        failures = ((failed_stream, error_class),)
        return startup_outcomes(identities, failures), ()
    return None, tuple(plans)


def assemble_sessions(
    plans: Sequence[CameraPlan],
    detector: DetectorProtocol,
    *,
    backend_client_factory: BackendClientFactory,
    source_factory: SourceFactory,
    evidence_max_bytes: int,
    delivery_interval_seconds: float,
    experimental_model_profile: ExperimentalPpeProfile | None = None,
    observation_schema_version: Literal["1.0.0", "1.1.0"] = "1.0.0",
    preview_observers: dict[str, CameraPreviewObserver] | None = None,
) -> tuple[CameraSession, ...]:
    """Create isolated tracker, poller, outbox, and evidence state per camera."""

    if not 1 <= len(plans) <= 3:
        raise CameraRuntimeError("runtime requires 1 to 3 cameras")
    if preview_observers is not None and (
        not isinstance(preview_observers, dict) or preview_observers
    ):
        raise CameraRuntimeError("Preview registry must be an empty per-runtime dictionary")
    if experimental_model_profile not in (None, "native-ppe-10", "unified-ppe-15"):
        raise CameraRuntimeError("unsupported experimental PPE profile")
    if observation_schema_version not in ("1.0.0", "1.1.0"):
        raise CameraRuntimeError("unsupported observation schema version")
    if experimental_model_profile is not None and observation_schema_version != "1.1.0":
        raise CameraRuntimeError("expanded PPE requires a configured v1.1.0 consumer")
    profile = experimental_ppe_profile(experimental_model_profile)
    sessions: list[CameraSession] = []
    for plan in plans:
        camera = plan.bound
        tracked_options = camera.entry.tracked_v17
        tracked_capture = None
        if tracked_options is not None:
            if (
                plan.generation is None
                or camera.evidence_dir is None
                or not camera.source_config.is_live
            ):
                raise CameraRuntimeError(
                    "Tracked v1.7 requires a live generation and evidence storage"
                )
            tracked_capture = TrackedCapturePolicy(
                credential=plan.generation,
                site_id=str(tracked_options.site_id),
                profile_sha256=tracked_options.profile_sha256,
                pending_seconds=tracked_options.pending_seconds,
                predecessor_wait_seconds=tracked_options.predecessor_wait_seconds,
                max_pending=tracked_options.max_pending,
                max_bytes=tracked_options.max_bytes,
            )
        evidence_publisher = None
        if camera.evidence_dir is not None:
            evidence_publisher = LocalEvidencePublisher(
                root_dir=camera.evidence_dir,
                max_jpeg_bytes=evidence_max_bytes,
                bind_content_hash=camera.entry.partitioned_frame_delivery
                or tracked_capture is not None,
            )
        pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline.for_model_profile(profile),
            zones=RestrictedZonePipeline(),
            ppe_region_id=str(camera.entry.ppe_region_id),
            schema_version=observation_schema_version,
            tracked_v17=tracked_capture is not None,
        )
        outbox = SqliteEventOutbox(camera.outbox)
        delivery_backend = backend_client_factory()
        source = source_factory(camera.source_config)
        if plan.generation is not None:
            plan.generation.ensure_current(datetime.now(UTC))
            source = GenerationBoundSource(source, plan.generation)
            plan.poller.enable_generation_ack(source)
        preview_queue = PreviewFrameQueue() if preview_observers is not None else None
        stream = StreamWorker(
            config=camera.source_config, source=source, preview_frames=preview_queue
        )
        preview = None
        analysis = None
        if preview_queue is not None:

            async def current_analysis(
                item: AnalysisFrame, stream=stream, store=plan.store, poller=plan.poller
            ) -> bool:
                try:
                    poller.ensure_fresh()
                except Exception:
                    return False
                current = await store.get_active_configuration(item.frame.camera_external_id)
                return current is item.configuration and stream.owns_current_frame(item.frame)

            analysis = AnalysisPreviewObserver(camera.camera_external_id, current=current_analysis)
            preview = CameraPreviewObserver(
                camera.camera_external_id,
                preview_queue,
                source_kind="LIVE" if camera.source_config.is_live else "REPLAY",
                is_streaming=lambda stream=stream: stream.state == StreamState.STREAMING,
                analysis=analysis,
            )
            preview_observers[camera.camera_external_id] = preview
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=detector,
            pipeline=pipeline,
            ppe_region_id=str(camera.entry.ppe_region_id),
            configurations=plan.store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(
                outbox,
                delivery_backend,
                generation_credential=plan.generation
                if tracked_capture is not None or camera.entry.generation_bound_event_delivery
                else None,
                tracked_batch_limit=tracked_capture.max_pending
                if tracked_capture is not None
                else None,
                tracked_evidence_upload=TrackedFrameEvidenceUploader(
                    delivery_backend, camera.evidence_dir
                )
                if camera.entry.private_evidence_upload and camera.evidence_dir is not None
                else None,
            ),
            tracked_capture=tracked_capture,
            analysis_observer=analysis.offer if analysis is not None else None,
            delivery_context_provider=plan.poller.snapshot_delivery_context
            if tracked_capture is not None or camera.entry.generation_bound_event_delivery
            else None,
            configuration_guard=plan.poller.ensure_fresh,
            delivery_interval_seconds=delivery_interval_seconds,
            evidence_publisher=evidence_publisher,
            partitioned_frame_delivery=camera.entry.partitioned_frame_delivery,
        )
        sessions.append(
            CameraSession(
                stream_id=camera.stream_id,
                camera_external_id=camera.camera_external_id,
                worker=worker,
                poller=plan.poller,
                preview=preview,
            )
        )
    return tuple(sessions)


async def close_configuration_clients(plans: Sequence[CameraPlan]) -> None:
    await _close_plans(plans)


async def _close_plans(plans: Sequence[CameraPlan]) -> None:
    for plan in plans:
        await plan.configuration_client.aclose()


async def _require_ppe_region(
    store: RegionConfigurationStore,
    camera_external_id: str,
    ppe_region_id: UUID,
) -> None:
    configuration = await store.get_active_configuration(camera_external_id)
    if configuration is None:
        raise CameraRuntimeError("Backend returned no active camera region configuration")
    if isinstance(configuration, (CameraConfigurationBundle, CameraPolicyConfigurationBundle)):
        return
    expected = str(ppe_region_id).casefold()
    if not any(region.region_id.casefold() == expected for region in configuration.regions):
        raise CameraRuntimeError(
            "configured PPE region is absent from the active camera configuration"
        )


__all__ = [
    "CameraPlan",
    "assemble_sessions",
    "close_configuration_clients",
    "preflight_manifest",
]
