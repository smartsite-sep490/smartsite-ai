"""Build per-camera MF05/MF06 sessions that share one detector."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker
from smartsite_ai.runtime.manifest import BoundCamera, BoundManifest, CameraRuntimeError
from smartsite_ai.runtime.supervisor import (
    CameraOutcome,
    CameraSession,
    _hold_cancellation,
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


async def preflight_manifest(
    bound: BoundManifest,
    *,
    configuration_client_factory: ConfigurationClientFactory,
) -> tuple[tuple[CameraOutcome, ...] | None, tuple[CameraPlan, ...]]:
    """Confirm every camera region before a model or source is opened."""

    plans: list[CameraPlan] = []
    identities = [(camera.stream_id, camera.camera_external_id) for camera in bound.cameras]
    try:
        for camera in bound.cameras:
            client = configuration_client_factory()
            try:
                store = RegionConfigurationStore()
                poller = CameraConfigurationPoller(
                    client=client,
                    store=store,
                    camera_id=camera.entry.camera_id,
                    camera_external_id=camera.camera_external_id,
                    poll_interval_seconds=bound.manifest.poll_interval_seconds,
                    stale_after_seconds=bound.manifest.stale_after_seconds,
                )
                await poller.refresh()
                poller.ensure_fresh()
                await _require_ppe_region(
                    store, camera.camera_external_id, camera.entry.ppe_region_id
                )
            except BaseException:
                async with _hold_cancellation():
                    await client.aclose()
                raise
            plans.append(
                CameraPlan(
                    bound=camera,
                    store=store,
                    poller=poller,
                    configuration_client=client,
                )
            )
    except asyncio.CancelledError:
        async with _hold_cancellation():
            await _close_plans(plans)
        raise
    except Exception as exc:
        await _close_plans(plans)
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
) -> tuple[CameraSession, ...]:
    """Create isolated tracker, poller, outbox, and evidence state per camera."""

    if not 1 <= len(plans) <= 3:
        raise CameraRuntimeError("runtime requires 1 to 3 cameras")
    sessions: list[CameraSession] = []
    for plan in plans:
        camera = plan.bound
        evidence_publisher = None
        if camera.evidence_dir is not None:
            evidence_publisher = LocalEvidencePublisher(
                root_dir=camera.evidence_dir,
                max_jpeg_bytes=evidence_max_bytes,
            )
        pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline(),
            zones=RestrictedZonePipeline(),
            ppe_region_id=str(camera.entry.ppe_region_id),
        )
        outbox = SqliteEventOutbox(camera.outbox)
        source = source_factory(camera.source_config)
        worker = HeadlessCameraProcessingWorker(
            stream=StreamWorker(config=camera.source_config, source=source),
            detector=detector,
            pipeline=pipeline,
            ppe_region_id=str(camera.entry.ppe_region_id),
            configurations=plan.store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend_client_factory()),
            configuration_guard=plan.poller.ensure_fresh,
            delivery_interval_seconds=delivery_interval_seconds,
            evidence_publisher=evidence_publisher,
        )
        sessions.append(
            CameraSession(
                stream_id=camera.stream_id,
                camera_external_id=camera.camera_external_id,
                worker=worker,
                poller=plan.poller,
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
    configuration = await store.get(camera_external_id)
    if configuration is None:
        raise CameraRuntimeError("Backend returned no active camera region configuration")
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
