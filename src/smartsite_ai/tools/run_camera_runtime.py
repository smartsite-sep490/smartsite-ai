"""Supervise 1–3 MF05/MF06 cameras with one verified YOLO11s load."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import UUID

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
from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
from smartsite_ai.runtime.inference_lane import share_detector
from smartsite_ai.runtime.manifest import (
    CameraRuntimeError,
    bind_manifest,
    load_manifest_bytes,
)
from smartsite_ai.runtime.media_host import run_with_camera_preview
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.supervisor import (
    CameraSession,
    RuntimePlan,
    _finish_cleanup,
    execute_runtime,
    exit_code,
    report_payload,
)
from smartsite_ai.source_resume_models import ResumeBinding, ResumeConfiguration, ResumeManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="absolute path of the 1-3 camera runtime manifest",
    )
    parser.add_argument("--generation-credential-file", type=Path)
    parser.add_argument("--expected-camera-id", type=UUID)
    parser.add_argument("--expected-outbox", type=Path)
    parser.add_argument(
        "--preview-port",
        type=int,
        help="explicit raw-preview HTTP port; requires SMARTSITE_AI_PREVIEW_SERVICE_TOKEN",
    )
    parser.add_argument("--require-resume-manifest", action="store_true")
    return parser


async def run_manifest(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    generation_credential_file: Path | None = None,
    expected_camera_id: UUID | None = None,
    expected_outbox: Path | None = None,
    preview_port: int | None = None,
    require_resume_manifest: bool = False,
) -> int:
    raw_expected = os.environ.get("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION")
    if require_resume_manifest and raw_expected is None:
        raise CameraRuntimeError("Expected resume manifest is required")
    expected_resume = None
    if raw_expected is not None:
        if len(raw_expected.encode("utf-8")) > 12_000:
            raise CameraRuntimeError("Expected resume manifest is too large")
        expected_resume = ResumeManifest.model_validate_json(raw_expected)
    supervised = any(
        value is not None
        for value in (generation_credential_file, expected_camera_id, expected_outbox)
    )
    runtime_settings = settings or (Settings(_env_file=None) if supervised else Settings())
    if preview_port is not None and (
        type(preview_port) is not int or not 1 <= preview_port <= 65535
    ):
        raise CameraRuntimeError("Preview port must be between 1 and 65535")
    if preview_port is not None and runtime_settings.preview_service_token is None:
        raise CameraRuntimeError("Preview service token must be configured")
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise CameraRuntimeError("manifest path must identify an absolute existing file")
    manifest = load_manifest_bytes(manifest_path.read_bytes())
    manifest = bind_generation_manifest(
        manifest, generation_credential_file, expected_camera_id, expected_outbox
    )
    bound = bind_manifest(manifest, os.environ)
    plans: list[CameraPlan] = []
    backend_clients: list[BackendClient] = []
    runner: UltralyticsYoloRunner | None = None
    observers: dict[str, CameraPreviewObserver] = {}

    async def preflight() -> tuple[object, ...] | None:
        failure, prepared = await preflight_manifest(
            bound,
            configuration_client_factory=lambda: BackendConfigurationClient.from_settings(
                runtime_settings
            ),
            policy_v2_client_factory=lambda: BackendConfigurationClient.from_settings(
                runtime_settings, policy_v2=True
            ),
        )
        plans.extend(prepared)
        if failure is None and expected_resume is not None:
            if len(prepared) != 1:
                raise CameraRuntimeError("Resume must bind exactly one tracked LIVE camera")
            plan = prepared[0]
            if (
                plan.generation is None
                or plan.bound.entry.tracked_v17 is None
                or not plan.bound.source_config.is_live
            ):
                raise CameraRuntimeError("Resume requires a tracked LIVE generation")
            actual_binding = ResumeBinding(
                cameraId=str(plan.bound.entry.camera_id),
                workerGenerationId=plan.generation.worker_generation_id,
                streamSessionId=plan.generation.stream_session_id,
            )
            bundle = await plan.store.get_bundle(plan.bound.camera_external_id)
            if (
                actual_binding != expected_resume.generationBinding
                or bundle is None
                or str(bundle.camera_id) != actual_binding.cameraId
            ):
                raise CameraRuntimeError("Expected resume generation binding changed")
            expected_resume.require_configuration(
                ResumeConfiguration(
                    configurationVersion=bundle.configuration_version,
                    viewRevision=bundle.view_revision,
                    bundleSha256=bundle.bundle_sha256,
                )
            )
        return failure

    async def open_model() -> DetectorProtocol:
        nonlocal runner
        detector, loaded_runner, _artifact, _class_map = load_yolo11_detector(
            bound.model_spec,
            runner_factory=UltralyticsYoloRunner,
            experimental_profile=bound.manifest.experimental_model_profile,
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
            experimental_model_profile=bound.manifest.experimental_model_profile,
            observation_schema_version=bound.manifest.observation_schema_version,
            preview_observers=observers if preview_port is not None else None,
        )

    async def close_clients() -> None:
        for client in backend_clients:
            await client.aclose()
        await close_configuration_clients(plans)

    try:
        plan = RuntimePlan(
            preflight=preflight,
            open_model=open_model,
            close_model=close_model,
            build_sessions=build_sessions,
        )
        if preview_port is None:
            report = await execute_runtime(plan)
        else:
            report = await run_with_camera_preview(
                plan,
                observers,
                runtime_settings.preview_service_token,
                host=str(runtime_settings.host),
                port=preview_port,
            )
    finally:
        await _finish_cleanup(close_clients())

    print(json.dumps(report_payload(report), separators=(",", ":")))
    return exit_code(report)


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preview_options = {} if args.preview_port is None else {"preview_port": args.preview_port}
    resume_options = {"require_resume_manifest": True} if args.require_resume_manifest else {}
    try:
        if any(
            value is not None
            for value in (
                args.generation_credential_file,
                args.expected_camera_id,
                args.expected_outbox,
            )
        ):
            return asyncio.run(
                run_manifest(
                    args.manifest,
                    generation_credential_file=args.generation_credential_file,
                    expected_camera_id=args.expected_camera_id,
                    expected_outbox=args.expected_outbox,
                    **preview_options,
                    **resume_options,
                )
            )
        return asyncio.run(run_manifest(args.manifest, **preview_options, **resume_options))
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
