"""Explicit single-camera operator composition; imports never start camera work."""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from smartsite_ai.configuration_runtime import ConfigurationCaptureRuntime
from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.integrations.backend_configuration_client import (
    BackendConfigurationClient,
    ConfigurationBundleResult,
)
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.runtime.manifest import load_manifest_bytes
from smartsite_ai.runtime.source_control_child import _require_live_source
from smartsite_ai.runtime.supervisor import _finish_cleanup
from smartsite_ai.source_activation import SourceActivator, SourceRegistry
from smartsite_ai.source_control_api import ProbeFrame
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_process_driver import SourceProcessDriver
from smartsite_ai.source_resume_models import ResumeConfiguration, ResumeManifest
from smartsite_ai.tools.configuration_host_config import HostConfiguration, load_host_configuration
from smartsite_ai.tools.run_camera_runtime import run_manifest
from smartsite_ai.tools.run_configuration_capture import _watch_stop_file, serve_host
from smartsite_ai.tools.run_source_control import (
    ActiveSource,
    HostManifest,
    build_source_control_app,
)
from smartsite_ai.training.dataset_integrity import DatasetIntegrityError, is_link_like


class OperatorProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    site_id: UUID = Field(alias="siteId")
    camera_id: UUID = Field(alias="cameraId")
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    host_manifest: Path = Field(alias="hostManifest")
    capture_manifest: Path = Field(alias="captureManifest")
    runtime_manifest: Path = Field(alias="runtimeManifest")
    state_directory: Path = Field(alias="stateDirectory")
    preview_port: int = Field(alias="previewPort", ge=1, le=65535)

    @field_validator("camera_external_id")
    @classmethod
    def printable_camera(cls, value: str) -> str:
        if not value.strip() or not value.isprintable():
            raise ValueError("Invalid camera external identifier")
        return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate operator configuration field")
    return result


def _local_path(path: Path, *, directory: bool = False) -> Path:
    if (
        not path.is_absolute()
        or str(path).startswith("\\\\")
        or path.resolve(strict=True) != path
        or any(parent.is_symlink() for parent in (path, *path.parents))
        or (not path.is_dir() if directory else not path.is_file())
    ):
        raise ValueError("Fixed existing local operator paths are required")
    return path


def _read_document(path: Path, maximum: int) -> bytes:
    _local_path(path)
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Operator configuration exceeds byte budget")
    json.loads(raw, object_pairs_hook=_unique_object)
    return raw


def load_operator_profile(path: Path) -> OperatorProfile:
    profile = OperatorProfile.model_validate_json(_read_document(path, 16_384))
    for linked in (profile.host_manifest, profile.capture_manifest, profile.runtime_manifest):
        _local_path(linked)
    _local_path(profile.state_directory, directory=True)
    return profile


async def probe_owned_image(
    origin: str,
    token: SecretStr,
    camera_external_id: str,
    generation_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Observe a fresh exact owner; never open a source or adopt a listening process."""
    address = urlsplit(origin)
    if (
        address.scheme != "http"
        or address.hostname not in {"127.0.0.1", "::1"}
        or not address.port
        or address.username is not None
        or address.password is not None
        or address.path not in {"", "/"}
        or address.query
        or address.fragment
    ):
        raise ValueError("Operator preview must be a fixed loopback origin")
    expected = str(UUID(generation_id))
    owned = client is None
    transport = client or httpx.AsyncClient(timeout=3, follow_redirects=False, trust_env=False)
    try:
        async with (
            asyncio.timeout(4),
            transport.stream(
                "GET",
                origin.rstrip("/") + "/v1/realtime/configuration-image",
                params={"cameraExternalId": camera_external_id},
                headers={"Authorization": "Bearer " + token.get_secret_value()},
                follow_redirects=False,
                timeout=3,
            ) as response,
        ):
            if (
                response.status_code != 200
                or response.headers.get("X-SmartSite-Worker-Generation") != expected
                or response.headers.get("content-type", "").split(";")[0] != "application/json"
            ):
                return False
            raw = bytearray()
            async for part in response.aiter_bytes():
                raw.extend(part)
                if len(raw) > 400_000:
                    return False
        value = json.loads(raw, object_pairs_hook=_unique_object)
        frame = value["frame"]
        checked = ProbeFrame.model_validate_json(
            json.dumps(
                {key: frame[key] for key in ("width", "height", "capturedAt", "imageDataUrl")}
            )
        )
        age = (datetime.now(UTC) - checked.capturedAt).total_seconds()
        return (
            value.get("schemaVersion") == 1
            and value.get("purpose") == "REGION_CONFIGURATION"
            and value.get("sourceKind") == "LIVE"
            and frame.get("cameraExternalId") == camera_external_id
            and checked.width * checked.height <= 8_000_000
            and -1 <= age <= 2
        )
    except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError):
        return False
    finally:
        if owned:
            await transport.aclose()


def _secret(environ: Mapping[str, str], name: str) -> SecretStr:
    value = environ[name]
    if not 32 <= len(value) <= 4096 or not value.isascii() or any(c.isspace() for c in value):
        raise ValueError("Operator server credential is unavailable or invalid")
    return SecretStr(value)


def _load_bindings(
    profile: OperatorProfile, environ: Mapping[str, str], *, selected: bool = False
) -> tuple[HostManifest, HostConfiguration]:
    scope = (str(profile.site_id), profile.camera_external_id)
    host = HostManifest.model_validate_json(_read_document(profile.host_manifest, 65_536))
    capture_document = json.loads(_read_document(profile.capture_manifest, 16_384))
    if (
        capture_document.get("sourceEnv") != "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"
        or capture_document.get("tokenEnv") != "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"
    ):
        raise ValueError("Capture must use the fixed operator environment bindings")
    # Validate capture configuration without opening a device or needing a selected source yet.
    capture = load_host_configuration(
        profile.capture_manifest,
        environ if selected else {**environ, "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0"},
        controlled_stop=True,
    )
    runtime = load_manifest_bytes(_read_document(profile.runtime_manifest, 65_536))
    if len(runtime.cameras) != 1:
        raise ValueError("Operator must bind exactly one tracked LIVE camera")
    camera = runtime.cameras[0]
    if (
        scope not in host.scopes
        and (scope[0], "*") not in host.scopes
        or capture.stream.camera_external_id != profile.camera_external_id
        or capture.source_kind != "LIVE"
        or capture.port != profile.preview_port
        or camera.camera_id != profile.camera_id
        or camera.camera_external_id != profile.camera_external_id
        or not camera.live
        or camera.tracked_v17 is None
        or camera.tracked_v17.site_id != profile.site_id
    ):
        raise ValueError("Operator manifests disagree on Camera, Site, source or preview scope")
    return host, capture


def build_operator_app(
    profile_path: Path, environ: Mapping[str, str]
) -> tuple[FastAPI, SourceProcessDriver]:
    """Compose owned control only. User commands explicitly start capture/inference later."""
    profile = load_operator_profile(profile_path)
    scope = (str(profile.site_id), profile.camera_external_id)
    control_token = _secret(environ, "SMARTSITE_AI_SOURCE_CONTROL_TOKEN")
    preview_token = _secret(environ, "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN")
    backend_token = _secret(environ, "SMARTSITE_AI_BACKEND_SERVICE_TOKEN")
    registry_key = _secret(environ, "SMARTSITE_AI_SOURCE_REGISTRY_KEY")
    backend_url = environ["SMARTSITE_AI_BACKEND_INGESTION_URL"]
    # Constructor validates the canonical URL without creating an HTTP client or socket.
    BackendConfigurationClient(backend_url, backend_token, policy_v2=True)
    host, capture = _load_bindings(profile, environ)
    origin = f"http://127.0.0.1:{profile.preview_port}"

    async def ready(generation_id: str) -> bool:
        return await probe_owned_image(
            origin, preview_token, profile.camera_external_id, generation_id
        )

    async def load_configuration(
        expected: ResumeManifest, _generation: Mapping[str, str]
    ) -> ResumeConfiguration:
        client = BackendConfigurationClient(backend_url, backend_token, policy_v2=True)
        try:
            fetched = await client.fetch_bundle(
                profile.camera_id, expected_camera_external_id=profile.camera_external_id
            )
            if not isinstance(fetched, ConfigurationBundleResult):
                raise ValueError("Accepted camera configuration is unavailable")
            return ResumeConfiguration(
                configurationVersion=fetched.bundle.configuration_version,
                viewRevision=fetched.bundle.view_revision,
                bundleSha256=fetched.bundle.bundle_sha256,
            )
        finally:
            await client.aclose()

    forwarded = {
        name: value
        for name, value in environ.items()
        if name
        in {
            "PATH",
            "Path",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN",
            "SMARTSITE_AI_BACKEND_SERVICE_TOKEN",
            "SMARTSITE_AI_BACKEND_INGESTION_URL",
            "SMARTSITE_AI_REALTIME_DEVICE",
        }
    }
    command = (
        str(Path(sys.executable)),
        "-m",
        "smartsite_ai.tools.source_operator",
        "--profile",
        str(profile_path),
        "--child",
    )
    driver = SourceProcessDriver(
        scope=scope,
        command=command,
        resume_command=command,
        stop_file=profile.state_directory / "STOP",
        generation_file=profile.state_directory / "generation.json",
        environment=forwarded,
        ready=ready,
        allow_unowned_start=True,
        resume_camera_id=str(profile.camera_id),
        resume_manifest_supported=True,
        resume_configuration_loader=load_configuration,
        capture_lease_directory=capture.lock_directory,
        capture_lease_resource_id=capture.resource_id,
    )
    registry = SourceRegistry(profile.state_directory / "source-registry.enc", registry_key)
    # Only the configured operator camera is served even if its host policy covers more Sites.
    scoped_host = host.model_copy(
        update={
            "scopes": [scope],
            "active_sources": [
                ActiveSource(
                    site_id=scope[0],
                    camera_external_id=scope[1],
                    source=SecretStr("0"),
                    preview_origin=origin,
                )
            ],
        }
    )
    app = build_source_control_app(
        scoped_host,
        control_token,
        activator=SourceActivator(registry, driver),
        owned_source=lambda requested: driver.current_source if requested == scope else None,
        preview_service_token=preview_token,
    )
    return app, driver


_OUTCOME_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_OUTCOME_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_OUTCOME_STATUSES = frozenset({"completed", "failed", "not_started"})


def _sanitize_runtime_outcome(
    payload: Mapping[str, object], *, expected_camera_external_id: str | None = None
) -> dict[str, object] | None:
    cameras = payload.get("cameras")
    if isinstance(cameras, list) and len(cameras) <= 3:
        sanitized_cameras: list[dict[str, object]] = []
        for entry in cameras:
            if not isinstance(entry, Mapping):
                return None
            status = entry.get("status")
            if not isinstance(status, str) or status not in _OUTCOME_STATUSES:
                return None
            item: dict[str, object] = {"status": status}
            stream_id = entry.get("streamId")
            if isinstance(stream_id, str) and _OUTCOME_ID_RE.fullmatch(stream_id):
                item["streamId"] = stream_id
            camera_external_id = entry.get("cameraExternalId")
            if (
                isinstance(camera_external_id, str)
                and _OUTCOME_ID_RE.fullmatch(camera_external_id)
                and (
                    expected_camera_external_id is None
                    or camera_external_id == expected_camera_external_id
                )
            ):
                item["cameraExternalId"] = camera_external_id
            for count_key in ("framesProcessed", "eventsEnqueued"):
                val = entry.get(count_key)
                if type(val) is int and val >= 0:
                    item[count_key] = val
            outbox = entry.get("outbox")
            if isinstance(outbox, Mapping):
                clean_outbox = {
                    key: outbox[key]
                    for key in ("pending", "delivered", "terminal")
                    if type(outbox.get(key)) is int and outbox[key] >= 0
                }
                if len(clean_outbox) == 3:
                    item["outbox"] = clean_outbox
            error_class = entry.get("errorClass")
            if isinstance(error_class, str) and _OUTCOME_IDENT_RE.fullmatch(error_class):
                item["errorClass"] = error_class
            sanitized_cameras.append(item)
        result: dict[str, object] = {"cameras": sanitized_cameras}
        for flag in ("modelLoaded", "modelClosed"):
            val = payload.get(flag)
            if type(val) is bool:
                result[flag] = val
        return result
    error_class = payload.get("errorClass")
    if error_class is None:
        error_class = payload.get("error")
    if isinstance(error_class, str) and _OUTCOME_IDENT_RE.fullmatch(error_class):
        return {"errorClass": error_class}
    return None


def _validate_runtime_outcome_path(path: Path) -> Path:
    if (
        not path.is_absolute()
        or str(path).startswith("\\\\")
        or not path.parent.is_dir()
        or path.parent.resolve(strict=True) != path.parent
        or path.is_symlink()
    ):
        raise ValueError("Invalid runtime outcome path")
    try:
        link_detected = any(is_link_like(parent) for parent in path.parents) or (
            os.path.lexists(path) and (is_link_like(path) or path.is_dir())
        )
    except DatasetIntegrityError as error:
        raise ValueError("Invalid runtime outcome path") from error
    if link_detected:
        raise ValueError("Invalid runtime outcome path")
    return path


def _write_runtime_outcome(
    path: Path, payload: Mapping[str, object], *, expected_camera_external_id: str | None = None
) -> None:
    _validate_runtime_outcome_path(path)
    sanitized = _sanitize_runtime_outcome(
        payload, expected_camera_external_id=expected_camera_external_id
    )
    if sanitized is None:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:12]}.tmp")
    data = (json.dumps(sanitized, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    fd: int | None = None
    created = False
    try:
        fd = os.open(
            tmp,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        created = True
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.close(fd)
        fd = None
        os.replace(tmp, path)
    except BaseException:
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
        if created:
            with suppress(OSError):
                tmp.unlink(missing_ok=True)
        raise


async def run_operator_child(profile_path: Path, environ: Mapping[str, str] | None = None) -> int:
    """Own capture first, then inference only for an accepted exact resume binding."""
    values = dict(os.environ if environ is None else environ)
    profile = load_operator_profile(profile_path)
    host, capture = _load_bindings(profile, values, selected=True)
    source = values["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"]
    if not source or len(source) > 4096:
        raise ValueError("Invalid selected source")
    _require_live_source(source, SourcePolicy(host.webcams, host.videos, host.networks))
    raw = values["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"]
    if len(raw.encode("utf-8")) > 4096:
        raise ValueError("Invalid generation binding")
    generation = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(generation, dict) or set(generation) != {
        "workerGenerationId",
        "streamSessionId",
        "secret",
        "expiresAt",
    }:
        raise ValueError("Invalid generation binding")
    credential = WorkerGenerationCredential.model_validate(
        {"schemaVersion": "1.0.0", "cameraId": str(profile.camera_id), **generation}
    )
    credential.ensure_current(datetime.now(UTC))
    raw_resume = values.get("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION")
    if raw_resume is not None:
        if len(raw_resume.encode("utf-8")) > 12_000:
            raise ValueError("Invalid accepted configuration binding")
        expected = ResumeManifest.model_validate_json(raw_resume)
        if (
            expected.generationBinding.cameraId != str(profile.camera_id)
            or expected.generationBinding.workerGenerationId != credential.worker_generation_id
            or expected.generationBinding.streamSessionId != credential.stream_session_id
        ):
            raise ValueError("Accepted configuration generation changed")
        BackendConfigurationClient(
            values.get("SMARTSITE_AI_BACKEND_INGESTION_URL"),
            _secret(values, "SMARTSITE_AI_BACKEND_SERVICE_TOKEN"),
            policy_v2=True,
        )
    stop_file = profile.state_directory / "STOP"
    if stop_file.is_symlink() or stop_file.exists():
        raise ValueError("A fresh owned stop path is required")
    outcome_path = (
        _validate_runtime_outcome_path(profile.state_directory / "runtime-outcome.json")
        if raw_resume is not None
        else None
    )
    outcome_recorded = False

    def record_outcome(payload: Mapping[str, object]) -> None:
        nonlocal outcome_recorded
        assert outcome_path is not None
        _write_runtime_outcome(
            outcome_path,
            payload,
            expected_camera_external_id=profile.camera_external_id,
        )
        outcome_recorded = True

    requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = {}
    watcher = None
    lease = None
    runtime_task = None
    release_proved = False
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is not None:
                previous[number] = signal.signal(
                    number, lambda _number, _frame: loop.call_soon_threadsafe(requested.set)
                )
        if raw_resume is not None:
            lease = ConfigurationProcessLease(capture.lock_directory, capture.resource_id)
            lease.acquire()
            assert outcome_path is not None
            with suppress(OSError):
                outcome_path.unlink(missing_ok=True)
        watcher = asyncio.create_task(_watch_stop_file(stop_file, requested))
        if raw_resume is None:
            result = await serve_host(
                capture,
                runtime_factory=lambda config: ConfigurationCaptureRuntime(
                    stream=config.stream,
                    source_kind=config.source_kind,
                    resource_id=config.resource_id,
                    lock_directory=config.lock_directory,
                    service_token=config.token,
                    worker_generation_id=credential.worker_generation_id,
                ),
                stop_requested=requested,
            )
            return 0 if result.get("nativeReleaseProved") is True else 1
        # This is a dedicated child process, not a shared HTTP request handler.
        # Bind only reviewed source-control variables for the existing CLI adapter.
        keys = (
            "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE",
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION",
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION",
            "SMARTSITE_AI_BACKEND_INGESTION_URL",
            "SMARTSITE_AI_BACKEND_SERVICE_TOKEN",
            "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN",
            "SMARTSITE_AI_REALTIME_DEVICE",
            "SMARTSITE_AI_HOST",
        )
        prior = {name: os.environ.get(name) for name in keys}
        try:
            for name in keys:
                if name in values:
                    os.environ[name] = values[name]
                else:
                    os.environ.pop(name, None)
            os.environ["SMARTSITE_AI_HOST"] = "127.0.0.1"
            runtime_task = asyncio.create_task(
                run_manifest(
                    profile.runtime_manifest,
                    expected_camera_id=profile.camera_id,
                    preview_port=profile.preview_port,
                    require_resume_manifest=True,
                    source_control_profile=profile.host_manifest,
                    report_callback=record_outcome,
                )
            )
            stopped = asyncio.create_task(requested.wait())
            try:
                done, _ = await asyncio.wait(
                    {runtime_task, stopped}, return_when=asyncio.FIRST_COMPLETED
                )
                if runtime_task in done:
                    code = runtime_task.result()
                    release_proved = code == 0
                    return code
                runtime_task.cancel()
                await _finish_cleanup(asyncio.gather(runtime_task, return_exceptions=True))
                if not runtime_task.cancelled():
                    code = runtime_task.result()
                    release_proved = code == 0
                    return code
                release_proved = True
                return 0
            finally:
                stopped.cancel()
                with suppress(asyncio.CancelledError):
                    await stopped
                if not runtime_task.done():
                    runtime_task.cancel()
                    await _finish_cleanup(asyncio.gather(runtime_task, return_exceptions=True))
        finally:
            for name, value in prior.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    except Exception as exc:
        if outcome_path is not None and not outcome_recorded:
            with suppress(Exception):
                _write_runtime_outcome(
                    outcome_path,
                    {"errorClass": type(exc).__name__},
                    expected_camera_external_id=profile.camera_external_id,
                )
        raise
    finally:
        try:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
        finally:
            try:
                if lease is not None and release_proved:
                    lease.release()
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)


async def _serve_operator(server: uvicorn.Server, driver: SourceProcessDriver) -> None:
    try:
        await server.serve()
    finally:
        if driver.process is not None:
            await driver.stop(driver.scope)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--port", type=int, default=62035)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.child:
            return asyncio.run(run_operator_child(args.profile))
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid control port")
        app, driver = build_operator_app(args.profile, os.environ)
        if args.port == load_operator_profile(args.profile).preview_port:
            raise ValueError("Control and preview must use distinct ports")
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=args.port,
                log_level="error",
                access_log=False,
            )
        )
        asyncio.run(_serve_operator(server, driver))
        return 0
    except (Exception, KeyboardInterrupt):
        print(json.dumps({"state": "SOURCE_OPERATOR_FAILED", "code": "OPERATOR_UNAVAILABLE"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
