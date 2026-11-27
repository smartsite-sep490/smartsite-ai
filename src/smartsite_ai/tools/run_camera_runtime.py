"""Supervise 1–3 MF05/MF06 cameras with one verified YOLO11s load."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from smartsite_ai.config import Settings
from smartsite_ai.inference.loading import load_yolo11_detector, load_yolo11_person_detector
from smartsite_ai.inference.person_anchored import PersonAnchoredPpeDetector
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.ingestion.browser_uplink import BrowserUplinkFrameSource
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.runtime.assembly import (
    CameraPlan,
    assemble_sessions,
    close_configuration_clients,
    preflight_manifest,
)
from smartsite_ai.runtime.browser_reader_assignment import load_browser_reader_assignment
from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
from smartsite_ai.runtime.inference_lane import SerializingDetector, share_detector
from smartsite_ai.runtime.manifest import (
    BrowserCameraSource,
    CameraRuntimeError,
    bind_manifest,
    load_manifest_bytes,
)
from smartsite_ai.runtime.media_host import run_with_camera_preview
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.source_control_child import (
    GENERATION_ENV,
    SOURCE_ENV,
    bind_source_control_child,
    bind_source_control_child_memory,
)
from smartsite_ai.runtime.supervisor import (
    CameraSession,
    RuntimePlan,
    RuntimeReport,
    _finish_cleanup,
    execute_runtime,
    exit_code,
    report_payload,
)
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_resume_models import ResumeBinding, ResumeConfiguration, ResumeManifest
from smartsite_ai.tools.run_source_control import HostManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="absolute path of the 1-3 camera runtime manifest",
    )
    parser.add_argument("--generation-credential-file", type=Path)
    parser.add_argument("--browser-reader-assignment-file", type=Path)
    parser.add_argument("--expected-camera-id", type=UUID)
    parser.add_argument("--expected-outbox", type=Path)
    parser.add_argument(
        "--preview-port",
        type=int,
        help="explicit raw-preview HTTP port; requires SMARTSITE_AI_PREVIEW_SERVICE_TOKEN",
    )
    parser.add_argument("--require-resume-manifest", action="store_true")
    parser.add_argument("--source-control-profile", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify Backend camera configuration and warm up the model without opening cameras",
    )
    return parser


def _synthetic_warmup_frame(camera_external_id: str, stream_id: str) -> FrameEnvelope:
    """Construct a synthetic 640x480 black BGR24 frame discarded after pre-capture warmup.

    Matching the 4:3 VGA source aspect avoids a rectangular letterbox shape recompile
    on the first real camera frame before CUDA/provider caches are warm.
    """
    return FrameEnvelope(
        stream_id=stream_id,
        session_id=UUID(int=0),
        camera_external_id=camera_external_id,
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        width=640,
        height=480,
        sequence_number=0,
        pixel_format="BGR24",
        payload=bytes(640 * 480 * 3),
    )


async def run_manifest(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    generation_credential_file: Path | None = None,
    browser_reader_assignment_file: Path | None = None,
    expected_camera_id: UUID | None = None,
    expected_outbox: Path | None = None,
    preview_port: int | None = None,
    require_resume_manifest: bool = False,
    source_control_profile: Path | None = None,
    check_only: bool = False,
    report_callback: Callable[[dict[str, object]], None] | None = None,
) -> int:
    if report_callback is not None and not callable(report_callback):
        raise TypeError("report_callback must be callable")
    try:
        report = await _run_manifest_report(
            manifest_path,
            settings=settings,
            generation_credential_file=generation_credential_file,
            browser_reader_assignment_file=browser_reader_assignment_file,
            expected_camera_id=expected_camera_id,
            expected_outbox=expected_outbox,
            preview_port=preview_port,
            require_resume_manifest=require_resume_manifest,
            source_control_profile=source_control_profile,
            check_only=check_only,
        )
    except Exception as exc:
        if report_callback is not None:
            with suppress(Exception):
                report_callback({"errorClass": type(exc).__name__})
        raise
    payload = report_payload(report)
    if check_only:
        payload["mode"] = "check"
    if report_callback is not None:
        with suppress(Exception):
            report_callback(payload)
    print(json.dumps(payload, separators=(",", ":")))
    if check_only:
        return (
            0
            if report.model_closed
            and report.cameras
            and all(camera.status == "configuration_checked" for camera in report.cameras)
            else 1
        )
    return exit_code(report)


async def _run_manifest_report(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    generation_credential_file: Path | None = None,
    browser_reader_assignment_file: Path | None = None,
    expected_camera_id: UUID | None = None,
    expected_outbox: Path | None = None,
    preview_port: int | None = None,
    require_resume_manifest: bool = False,
    source_control_profile: Path | None = None,
    check_only: bool = False,
) -> RuntimeReport:
    if check_only and (preview_port is not None or source_control_profile is not None):
        raise CameraRuntimeError("Check mode cannot start preview or bind a source-control child")
    if source_control_profile is None and any(
        key in os.environ for key in (SOURCE_ENV, GENERATION_ENV)
    ):
        raise CameraRuntimeError(
            "Runtime source credentials require an explicit source-control profile"
        )
    raw_expected = os.environ.get("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION")
    if (require_resume_manifest or source_control_profile is not None) and raw_expected is None:
        raise CameraRuntimeError("Expected resume manifest is required")
    expected_resume = None
    if raw_expected is not None:
        if len(raw_expected.encode("utf-8")) > 12_000:
            raise CameraRuntimeError("Expected resume manifest is too large")
        expected_resume = ResumeManifest.model_validate_json(raw_expected)
    supervised = any(
        value is not None
        for value in (
            generation_credential_file,
            browser_reader_assignment_file,
            expected_camera_id,
            expected_outbox,
            source_control_profile,
        )
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
    browser_cameras = tuple(
        camera for camera in manifest.cameras if isinstance(camera.source, BrowserCameraSource)
    )
    browser_assignment = None
    if browser_cameras or browser_reader_assignment_file is not None:
        if (
            len(browser_cameras) != 1
            or len(manifest.cameras) != 1
            or browser_reader_assignment_file is None
            or source_control_profile is not None
        ):
            raise CameraRuntimeError("Browser webcam requires a single supervised assignment")
        browser_camera = browser_cameras[0]
        browser_assignment = load_browser_reader_assignment(browser_reader_assignment_file)
        if (
            browser_camera.tracked_v17 is None
            or browser_assignment.camera_id != browser_camera.camera_id
            or browser_assignment.site_id != browser_camera.tracked_v17.site_id
        ):
            raise CameraRuntimeError("Browser reader scope does not match the camera manifest")
    generation_override = None
    if source_control_profile is not None:
        if (
            expected_camera_id is None
            or generation_credential_file is not None
            or expected_outbox is not None
            or expected_resume is None
            or not source_control_profile.is_absolute()
            or not source_control_profile.is_file()
            or source_control_profile.resolve() != source_control_profile
        ):
            raise CameraRuntimeError("Source-control profile requires fixed reviewed runtime paths")
        with source_control_profile.open("rb") as profile_file:
            profile_bytes = profile_file.read(65_537)
        if len(profile_bytes) > 65_536:
            raise CameraRuntimeError("Source-control profile is too large")
        host = HostManifest.model_validate_json(profile_bytes)
        binding_options = {
            "policy": SourcePolicy(host.webcams, host.videos, host.networks),
            "scopes": set(host.scopes),
            "camera_id": expected_camera_id,
            "environ": os.environ,
            "expected": expected_resume,
        }
        if os.name == "nt":
            try:
                manifest, generation_override = bind_source_control_child_memory(
                    manifest, **binding_options
                )
            finally:
                # Do not inherit the new generation secret into further children.
                os.environ.pop(GENERATION_ENV, None)
        else:
            manifest = bind_source_control_child(manifest, **binding_options)
    else:
        manifest = bind_generation_manifest(
            manifest, generation_credential_file, expected_camera_id, expected_outbox
        )
    bound = bind_manifest(manifest, os.environ)
    plans: list[CameraPlan] = []
    backend_clients: list[BackendClient] = []
    runner: UltralyticsYoloRunner | None = None
    person_runner: UltralyticsYoloRunner | None = None
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
            generation_override=generation_override,
            **({"check_only": True} if check_only else {}),
        )
        plans.extend(prepared)
        if failure is None and browser_assignment is not None:
            generation = prepared[0].generation
            if generation is None or generation.stream_session_id != str(
                browser_assignment.lease_id
            ):
                raise CameraRuntimeError(
                    "Browser reader lease does not match the worker generation"
                )
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
        nonlocal runner, person_runner
        detector, loaded_runner, _artifact, _class_map = load_yolo11_detector(
            bound.model_spec,
            runner_factory=UltralyticsYoloRunner,
            experimental_profile=bound.manifest.experimental_model_profile,
        )
        loaded_person_runner = None
        try:
            if not isinstance(loaded_runner, UltralyticsYoloRunner):
                raise TypeError("runtime detector loader returned an unexpected runner")
            if bound.person_model_spec is not None:
                persons, loaded_person_runner, _, _ = load_yolo11_person_detector(
                    bound.person_model_spec, runner_factory=UltralyticsYoloRunner
                )
                if not isinstance(loaded_person_runner, UltralyticsYoloRunner):
                    raise TypeError("runtime person loader returned an unexpected runner")
                detector = PersonAnchoredPpeDetector(ppe=detector, persons=persons)
            shared = (
                SerializingDetector(detector)
                if loaded_person_runner is not None
                else share_detector(detector, loaded_runner)
            )
            warmup_camera = bound.cameras[0]
            await shared.detect(
                _synthetic_warmup_frame(warmup_camera.camera_external_id, warmup_camera.stream_id)
            )
        except BaseException:
            try:
                if loaded_person_runner is not None:
                    loaded_person_runner.close()
            finally:
                loaded_runner.close()
            raise
        runner = loaded_runner
        person_runner = loaded_person_runner
        return shared

    async def close_model() -> None:
        try:
            if person_runner is not None:
                person_runner.close()
        finally:
            if runner is not None:
                runner.close()

    def build_sessions(detector: DetectorProtocol) -> tuple[CameraSession, ...]:
        def backend_client_factory() -> BackendClient:
            client = BackendClient.from_settings(runtime_settings)
            backend_clients.append(client)
            return client

        def source_factory(config: StreamConfig) -> OpenCvFrameSource | BrowserUplinkFrameSource:
            assignment = browser_assignment
            if assignment is None:
                return OpenCvFrameSource(config)
            backend = backend_client_factory()

            async def read_packet(after_sequence: int) -> dict[str, object] | None:
                return await backend.read_browser_frame(
                    assignment.site_id,
                    assignment.camera_id,
                    assignment.lease_id,
                    after_sequence,
                    assignment.reader_secret,
                )

            return BrowserUplinkFrameSource.from_frame_reader(
                read_packet,
                stream_id=config.stream_id,
                camera_external_id=config.camera_external_id,
                session_id=assignment.lease_id,
            )

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
        preview_options = {}
        if source_control_profile is not None and expected_resume is not None:
            preview_options["worker_generation_ids"] = {
                bound.cameras[
                    0
                ].camera_external_id: expected_resume.generationBinding.workerGenerationId
            }
        if check_only:
            report = await execute_runtime(
                plan,
                check_cameras=tuple(
                    (camera.stream_id, camera.camera_external_id) for camera in bound.cameras
                ),
            )
        elif preview_port is None:
            report = await execute_runtime(plan)
        else:
            report = await run_with_camera_preview(
                plan,
                observers,
                runtime_settings.preview_service_token,
                host=str(runtime_settings.host),
                port=preview_port,
                **preview_options,
            )
    finally:
        await _finish_cleanup(close_clients())

    return report


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preview_options = {} if args.preview_port is None else {"preview_port": args.preview_port}
    resume_options = {"require_resume_manifest": True} if args.require_resume_manifest else {}
    if args.browser_reader_assignment_file is not None:
        resume_options["browser_reader_assignment_file"] = args.browser_reader_assignment_file
    if args.check:
        resume_options["check_only"] = True
    try:
        if args.source_control_profile is not None:
            return asyncio.run(
                run_manifest(
                    args.manifest,
                    source_control_profile=args.source_control_profile,
                    expected_camera_id=args.expected_camera_id,
                    generation_credential_file=args.generation_credential_file,
                    expected_outbox=args.expected_outbox,
                    **preview_options,
                    **resume_options,
                )
            )
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
