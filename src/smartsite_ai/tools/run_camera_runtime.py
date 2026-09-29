"""Supervise 1–3 MF05/MF06 cameras with one verified YOLO11s load."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from smartsite_ai.config import Settings
from smartsite_ai.inference.loading import load_yolo11_detector
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.runtime.assembly import (
    CameraPlan,
    assemble_sessions,
    close_configuration_clients,
    preflight_manifest,
)
from smartsite_ai.runtime.inference_lane import share_detector
from smartsite_ai.runtime.manifest import (
    CameraRuntimeError,
    bind_manifest,
    load_manifest_bytes,
)
from smartsite_ai.runtime.supervisor import (
    CameraSession,
    RuntimePlan,
    _hold_cancellation,
    execute_runtime,
    exit_code,
    report_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="absolute path of the 1-3 camera runtime manifest",
    )
    return parser


async def run_manifest(manifest_path: Path, *, settings: Settings | None = None) -> int:
    runtime_settings = settings or Settings()
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise CameraRuntimeError("manifest path must identify an absolute existing file")
    manifest = load_manifest_bytes(manifest_path.read_bytes())
    bound = bind_manifest(manifest, os.environ)
    plans: list[CameraPlan] = []
    backend_clients: list[BackendClient] = []
    runner: UltralyticsYoloRunner | None = None

    async def preflight() -> tuple[object, ...] | None:
        failure, prepared = await preflight_manifest(
            bound,
            configuration_client_factory=lambda: BackendConfigurationClient.from_settings(
                runtime_settings
            ),
        )
        plans.extend(prepared)
        return failure

    async def open_model() -> DetectorProtocol:
        nonlocal runner
        detector, loaded_runner, _artifact, _class_map = load_yolo11_detector(
            bound.model_spec,
            runner_factory=UltralyticsYoloRunner,
        )
        try:
            if not isinstance(loaded_runner, UltralyticsYoloRunner):
                raise TypeError("runtime detector loader returned an unexpected runner")
            shared = share_detector(detector, loaded_runner)
        except BaseException:
            loaded_runner.close()
            raise
        runner = loaded_runner
        return shared

    async def close_model() -> None:
        if runner is not None:
            runner.close()

    def build_sessions(detector: DetectorProtocol) -> tuple[CameraSession, ...]:
        def backend_client_factory() -> BackendClient:
            client = BackendClient.from_settings(runtime_settings)
            backend_clients.append(client)
            return client

        def source_factory(config: StreamConfig) -> OpenCvFrameSource:
            return OpenCvFrameSource(config)

        return assemble_sessions(
            plans,
            detector,
            backend_client_factory=backend_client_factory,
            source_factory=source_factory,
            evidence_max_bytes=bound.manifest.evidence_max_bytes,
            delivery_interval_seconds=bound.manifest.delivery_interval_seconds,
        )

    try:
        report = await execute_runtime(
            RuntimePlan(
                preflight=preflight,
                open_model=open_model,
                close_model=close_model,
                build_sessions=build_sessions,
            )
        )
    finally:
        async with _hold_cancellation():
            for client in backend_clients:
                await client.aclose()
            await close_configuration_clients(plans)

    print(json.dumps(report_payload(report), separators=(",", ":")))
    return exit_code(report)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_manifest(args.manifest))
    except KeyboardInterrupt:
        return 130
    except CameraRuntimeError as exc:
        print(
            json.dumps(
                {"error": "CameraRuntimeError", "message": str(exc)},
                separators=(",", ":"),
            )
        )
        return 1
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}, separators=(",", ":")))
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
