"""Run one MF05/MF06 camera or video pipeline independently of browser clients."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from smartsite_ai.config import Settings
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
from smartsite_ai.inference.loading import load_yolo11_detector
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker, ProcessingWorkerResult
from smartsite_ai.tracking.iou_tracker import IoUPersonTracker
from smartsite_ai.training.dataset_integrity import is_link_like


class CameraWorkerRunError(RuntimeError):
    """A safe operator-facing failure from the explicit worker lifecycle."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="absolute video path, camera index, or RTSP URL; prefer env for credentialed URLs",
    )
    parser.add_argument(
        "--live", action="store_true", help="reconnect source after live disconnect"
    )
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--camera-id", type=UUID, required=True, help="Backend camera UUID")
    parser.add_argument("--camera-external-id", required=True)
    parser.add_argument("--ppe-region-id", type=UUID, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--stale-after", type=float, default=60.0)
    parser.add_argument("--delivery-interval", type=float, default=0.25)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="root directory for local evidence storage (disabled by default)",
    )
    parser.add_argument(
        "--evidence-max-bytes",
        type=int,
        default=1_048_576,
        help="maximum size of single JPEG evidence image in bytes (default: 1MB)",
    )
    return parser


def _absolute_existing_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CameraWorkerRunError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise CameraWorkerRunError(f"{label} path must identify a regular non-symlink file")
    return path


def _absolute_existing_dir(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CameraWorkerRunError(f"{label} path must be absolute")
    if not path.is_dir() or is_link_like(path):
        raise CameraWorkerRunError(
            f"{label} path must identify an existing regular non-symlink directory"
        )
    return path


def _absolute_outbox_path(path: Path) -> Path:
    if not path.is_absolute():
        raise CameraWorkerRunError("outbox path must be absolute")
    if not path.parent.is_dir():
        raise CameraWorkerRunError("outbox parent directory must already exist")
    return path


def _resolve_source(cli_source: str | None, settings: Settings) -> str:
    source = cli_source
    if source is None and settings.worker_source is not None:
        source = settings.worker_source.get_secret_value()
    if source is None or not source.strip():
        raise CameraWorkerRunError(
            "camera source must be provided by --source or SMARTSITE_AI_WORKER_SOURCE"
        )
    return source


async def _require_ppe_region(
    store: RegionConfigurationStore,
    camera_external_id: str,
    ppe_region_id: UUID,
) -> None:
    configuration = await store.get(camera_external_id)
    if configuration is None:
        raise CameraWorkerRunError("Backend returned no active camera region configuration")
    expected = str(ppe_region_id).casefold()
    if not any(region.region_id.casefold() == expected for region in configuration.regions):
        raise CameraWorkerRunError(
            "configured PPE region is absent from the active camera configuration"
        )


async def _coordinate(
    worker: HeadlessCameraProcessingWorker,
    poller: CameraConfigurationPoller,
) -> ProcessingWorkerResult:
    stop_polling = asyncio.Event()
    worker_task = asyncio.create_task(worker.run(), name="headless-camera-processing")
    poller_task = asyncio.create_task(poller.run(stop_polling), name="camera-configuration-poll")
    try:
        done, _pending = await asyncio.wait(
            (worker_task, poller_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if poller_task in done:
            poller_task.result()
            raise CameraWorkerRunError("configuration poller stopped unexpectedly")
        result = worker_task.result()
        stop_polling.set()
        await poller_task
        return result
    finally:
        stop_polling.set()
        for task in (worker_task, poller_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(worker_task, poller_task, return_exceptions=True)


async def run_worker(args: argparse.Namespace, *, settings: Settings | None = None) -> int:
    runtime_settings = settings or Settings()
    model_spec_path = _absolute_existing_file(args.model_spec, "model spec")
    outbox_path = _absolute_outbox_path(args.outbox)
    source = _resolve_source(args.source, runtime_settings)

    source_config = StreamConfig(
        stream_id=args.stream_id,
        camera_external_id=args.camera_external_id,
        source_url=source,
        target_fps=args.target_fps,
        is_live=args.live,
        pace_replay=not args.live,
        max_consecutive_failures=None if args.live else 1,
    )
    store = RegionConfigurationStore()
    runner: UltralyticsYoloRunner | None = None
    async with (
        BackendConfigurationClient.from_settings(runtime_settings) as configuration_client,
        BackendClient.from_settings(runtime_settings) as backend_client,
    ):
        poller = CameraConfigurationPoller(
            client=configuration_client,
            store=store,
            camera_id=args.camera_id,
            camera_external_id=args.camera_external_id,
            poll_interval_seconds=args.poll_interval,
            stale_after_seconds=args.stale_after,
        )
        # Validate authoritative regions before opening a camera or loading GPU state.
        await poller.refresh()
        poller.ensure_fresh()
        await _require_ppe_region(store, args.camera_external_id, args.ppe_region_id)

        detector, loaded_runner, _artifact, _class_map = load_yolo11_detector(
            model_spec_path,
            runner_factory=UltralyticsYoloRunner,
        )
        runner = loaded_runner
        try:
            frame_source = OpenCvFrameSource(source_config)
            stream = StreamWorker(config=source_config, source=frame_source)
            outbox = SqliteEventOutbox(outbox_path)
            pipeline = Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline(),
                zones=RestrictedZonePipeline(),
                ppe_region_id=str(args.ppe_region_id),
            )
            evidence_publisher = None
            if args.evidence_dir is not None:
                evidence_dir = _absolute_existing_dir(args.evidence_dir, "evidence-dir")
                evidence_publisher = LocalEvidencePublisher(
                    root_dir=evidence_dir,
                    max_jpeg_bytes=args.evidence_max_bytes,
                )

            worker = HeadlessCameraProcessingWorker(
                stream=stream,
                detector=detector,
                pipeline=pipeline,
                ppe_region_id=str(args.ppe_region_id),
                configurations=store,
                outbox=outbox,
                dispatcher=OutboxDispatcher(outbox, backend_client),
                configuration_guard=poller.ensure_fresh,
                delivery_interval_seconds=args.delivery_interval,
                evidence_publisher=evidence_publisher,
            )
            result = await _coordinate(worker, poller)
        finally:
            runner.close()

    print(
        json.dumps(
            {
                "framesProcessed": result.frames_processed,
                "eventsEnqueued": result.events_enqueued,
                "outbox": {
                    "pending": result.outbox.pending,
                    "delivered": result.outbox.delivered,
                    "terminal": result.outbox.terminal,
                },
            },
            separators=(",", ":"),
        )
    )
    return 0 if result.outbox.pending == 0 and result.outbox.terminal == 0 else 2


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_worker(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}, separators=(",", ":")))
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
