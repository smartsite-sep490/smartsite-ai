"""Bounded RED tests for explicit AI source operator composition (profile, probe, factory)."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_tracked_runtime_selection import document

from smartsite_ai import configuration_runtime
from smartsite_ai.inference import loading as model_loading
from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.source_process_driver import SourceProcessDriver
from smartsite_ai.tools import (
    run_camera_runtime,
    run_configuration_capture,
)
from smartsite_ai.tools import run_source_control as stock_control

MODULE = "smartsite_ai.tools.source_operator"
GEN1 = "11111111-2222-4333-8444-555555555555"
GEN2 = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
CMD1 = "22222222-2222-4222-8222-222222222222"
HEADER = "X-SmartSite-Worker-Generation"
CONTROL_TOKEN = "control-" + "c" * 40
PREVIEW_TOKEN = "preview-" + "p" * 40
BACKEND_TOKEN = "backend-" + "b" * 40


def _op() -> ModuleType:
    assert importlib.util.find_spec(MODULE) is not None, "Operator composition module is missing"
    return importlib.import_module(MODULE)


def _valid_image_payload(
    camera_external_id: str,
    *,
    captured_at: datetime | None = None,
    source_kind: str = "LIVE",
    width: int = 640,
    height: int = 480,
    image_data_url: str = "data:image/jpeg;base64,/9j/2Q==",
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "purpose": "REGION_CONFIGURATION",
        "sourceKind": source_kind,
        "frame": {
            "cameraExternalId": camera_external_id,
            "sessionId": str(uuid4()),
            "sequenceNumber": "0",
            "capturedAt": (captured_at or datetime.now(UTC)).isoformat(),
            "width": width,
            "height": height,
            "imageDataUrl": image_data_url,
            "imageSha256": "0" * 64,
        },
    }


def _operator_bundle(
    tmp_path: Path,
    *,
    preview_port: int = 62036,
    host_scopes: list[list[str]] | None = None,
    capture_port: int | None = None,
    capture_camera: str | None = None,
    capture_source_env: str = "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE",
    profile_overrides: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, str]]:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    runtime_doc, bundle = document(runtime_dir)
    site_id = str(runtime_doc["cameras"][0]["trackedV17"]["siteId"])
    camera_id = str(bundle.camera_id)
    camera_ext = bundle.camera_external_id

    host_path = tmp_path / "host-manifest.json"
    host_path.write_text(
        json.dumps(
            {
                "host_id": "edge-op",
                "label": "Operator Edge",
                "scopes": host_scopes if host_scopes is not None else [[site_id, camera_ext]],
                "webcams": {"usb0": 0},
            }
        ),
        encoding="utf-8",
    )

    state_dir = tmp_path / "operator-state"
    state_dir.mkdir(exist_ok=True)
    lock_dir = state_dir / "locks"
    lock_dir.mkdir(exist_ok=True)

    capture_path = tmp_path / "capture-manifest.json"
    capture_path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "streamId": "configuration-only",
                "cameraExternalId": capture_camera or camera_ext,
                "sourceKind": "LIVE",
                "sourceEnv": capture_source_env,
                "tokenEnv": "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN",
                "resourceId": "e98f6475-0660-4c71-a466-4c93d1739ee0",
                "lockDirectory": str(lock_dir),
                "port": capture_port if capture_port is not None else preview_port,
                "durationSeconds": 60,
            }
        ),
        encoding="utf-8",
    )

    runtime_path = tmp_path / "runtime-manifest.json"
    runtime_path.write_text(json.dumps(runtime_doc), encoding="utf-8")

    profile_doc: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "siteId": site_id,
        "cameraId": camera_id,
        "cameraExternalId": camera_ext,
        "hostManifest": str(host_path),
        "captureManifest": str(capture_path),
        "runtimeManifest": str(runtime_path),
        "stateDirectory": str(state_dir),
        "previewPort": preview_port,
    }
    if profile_overrides:
        profile_doc.update(profile_overrides)
    profile_path = tmp_path / "operator-profile.json"
    profile_path.write_text(json.dumps(profile_doc), encoding="utf-8")

    environ = {
        "SMARTSITE_AI_SOURCE_CONTROL_TOKEN": CONTROL_TOKEN,
        "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN": PREVIEW_TOKEN,
        "SMARTSITE_AI_SOURCE_REGISTRY_KEY": Fernet.generate_key().decode(),
        "SMARTSITE_AI_BACKEND_INGESTION_URL": "https://backend.example/api/v1/integrations/ai/events",
        "SMARTSITE_AI_BACKEND_SERVICE_TOKEN": BACKEND_TOKEN,
    }
    return profile_path, profile_doc, environ


def test_load_operator_profile_validates_bounds_aliases_and_path_integrity(tmp_path: Path) -> None:
    op = _op()
    profile_path, doc, _ = _operator_bundle(tmp_path)
    loaded = op.load_operator_profile(profile_path)
    assert isinstance(loaded, op.OperatorProfile)
    assert loaded.schema_version == "1.0.0"
    assert str(loaded.site_id) == doc["siteId"]
    assert str(loaded.camera_id) == doc["cameraId"]
    assert loaded.camera_external_id == doc["cameraExternalId"]
    assert loaded.host_manifest == Path(str(doc["hostManifest"])).resolve()
    assert loaded.capture_manifest == Path(str(doc["captureManifest"])).resolve()
    assert loaded.runtime_manifest == Path(str(doc["runtimeManifest"])).resolve()
    assert loaded.state_directory == Path(str(doc["stateDirectory"])).resolve()
    assert loaded.preview_port == 62036

    with pytest.raises((ValueError, AttributeError)):
        loaded.preview_port = 1

    # Duplicate keys, >16KiB, unknown/credential/command keys, bad ports/IDs, missing paths reject.
    dup_path = tmp_path / "dup.json"
    raw = profile_path.read_text(encoding="utf-8")
    dup_path.write_text(raw[:-1] + ', "previewPort": 62036}', encoding="utf-8")
    with pytest.raises(ValueError):
        op.load_operator_profile(dup_path)

    oversize = tmp_path / "oversize.json"
    oversize.write_text(raw + (" " * 17_000), encoding="utf-8")
    with pytest.raises(ValueError):
        op.load_operator_profile(oversize)

    for bad in (
        {"unknownField": 1},
        {"source": "rtsp://u:p@10.0.0.1/cam"},
        {"command": ["/bin/true"]},
        {"schemaVersion": "1.1.0"},
        {"siteId": "not-a-uuid"},
        {"cameraExternalId": ""},
        {"cameraExternalId": "x" * 129},
        {"previewPort": 0},
        {"previewPort": 65536},
        {"previewPort": True},
        {"hostManifest": "relative/host.json"},
        {"runtimeManifest": str(tmp_path / "missing-runtime.json")},
        {"stateDirectory": str(profile_path)},
    ):
        bad_path = tmp_path / f"bad-{abs(hash(tuple(bad.keys())))}.json"
        bad_doc = {**doc, **bad}
        bad_path.write_text(json.dumps(bad_doc), encoding="utf-8")
        with pytest.raises((ValueError, OSError)):
            op.load_operator_profile(bad_path)

    symlink_profile = tmp_path / "link-profile.json"
    try:
        symlink_profile.symlink_to(profile_path)
    except (OSError, NotImplementedError):
        return
    with pytest.raises((ValueError, OSError)):
        op.load_operator_profile(symlink_profile)


@pytest.mark.anyio
async def test_probe_owned_image_verifies_header_freshness_bounds_and_client_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _op()
    cam = "camera-01"
    origin = "http://127.0.0.1:62036"

    def make_transport(handler):
        return httpx.MockTransport(handler)

    seen_auth: list[str] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        assert request.url.path == "/v1/realtime/configuration-image"
        assert request.url.params["cameraExternalId"] == cam
        assert PREVIEW_TOKEN not in str(request.url)
        return httpx.Response(
            200,
            headers={HEADER: GEN1},
            json=_valid_image_payload(cam),
        )

    caller_client = httpx.AsyncClient(transport=make_transport(ok_handler))
    try:
        assert (
            await op.probe_owned_image(
                origin, SecretStr(PREVIEW_TOKEN), cam, GEN1.upper(), client=caller_client
            )
            is True
        )
        assert not caller_client.is_closed
        assert seen_auth == [f"Bearer {PREVIEW_TOKEN}"]
    finally:
        await caller_client.aclose()

    # Rejection cases -> must return False and keep errors bounded/secret-free.
    now = datetime.now(UTC)
    failure_responses: list[object] = [
        httpx.Response(401, headers={HEADER: GEN1}, json={"code": "UNAUTHENTICATED"}),
        httpx.Response(503, headers={HEADER: GEN1}, json={"code": "UNAVAILABLE"}),
        httpx.Response(302, headers={"Location": "http://127.0.0.1:1/x", HEADER: GEN1}),
        httpx.Response(200, json=_valid_image_payload(cam)),  # missing generation header
        httpx.Response(200, headers={HEADER: GEN2}, json=_valid_image_payload(cam)),
        httpx.Response(
            200, headers={HEADER: GEN1}, json=_valid_image_payload(cam, source_kind="REPLAY")
        ),
        httpx.Response(200, headers={HEADER: GEN1}, json=_valid_image_payload("other-cam")),
        httpx.Response(
            200,
            headers={HEADER: GEN1},
            json=_valid_image_payload(cam, captured_at=now - timedelta(seconds=3)),
        ),
        httpx.Response(
            200,
            headers={HEADER: GEN1},
            json=_valid_image_payload(cam, captured_at=now + timedelta(seconds=2)),
        ),
        httpx.Response(
            200,
            headers={HEADER: GEN1},
            json=_valid_image_payload(cam, image_data_url="https://remote.example/x.jpg"),
        ),
        httpx.Response(
            200,
            headers={HEADER: GEN1},
            content=b'{"schemaVersion":1,"pad":"' + (b"x" * 400_050) + b'"}',
        ),
    ]
    for item in failure_responses:
        async with httpx.AsyncClient(
            transport=make_transport(lambda _req, resp=item: resp)
        ) as fail_client:
            assert (
                await op.probe_owned_image(
                    origin, SecretStr(PREVIEW_TOKEN), cam, GEN1, client=fail_client
                )
                is False
            )

    def timeout_handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timeout token={PREVIEW_TOKEN}")

    async with httpx.AsyncClient(transport=make_transport(timeout_handler)) as timeout_client:
        assert (
            await op.probe_owned_image(
                origin, SecretStr(PREVIEW_TOKEN), cam, GEN1, client=timeout_client
            )
            is False
        )

    for bad_origin in (
        "https://remote.example:62036",
        "http://0.0.0.0:62036",
        "http://user:pass@127.0.0.1:62036",
        "http://127.0.0.1:62036/path?x=1",
    ):
        try:
            res = await op.probe_owned_image(bad_origin, SecretStr(PREVIEW_TOKEN), cam, GEN1)
            assert res is False
        except ValueError:
            pass

    class _HangingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"schemaVersion":1,'
            await asyncio.sleep(5)

    real_timeout = op.asyncio.timeout
    monkeypatch.setattr(op.asyncio, "timeout", lambda _seconds: real_timeout(0.05))
    async with httpx.AsyncClient(
        transport=make_transport(
            lambda _req: httpx.Response(
                200,
                headers={"content-type": "application/json", HEADER: GEN1},
                stream=_HangingStream(),
            )
        )
    ) as hanging_client:
        assert (
            await op.probe_owned_image(
                origin, SecretStr(PREVIEW_TOKEN), cam, GEN1, client=hanging_client
            )
            is False
        )
        assert not hanging_client.is_closed


def test_build_operator_app_wires_driver_scope_redacted_env_and_default_cli_stays_disabled(
    tmp_path: Path,
) -> None:
    op = _op()
    profile_path, doc, environ = _operator_bundle(tmp_path)
    full_env = {
        **environ,
        "DATABASE_URL": "postgres://secret:db@localhost/smartsite",
        "PGPASSWORD": "db-password-never-forward",
        "PGUSER": "postgres",
    }

    # Missing required server variables reject cleanly before building app/driver.
    for required_key in list(environ):
        partial_env = {k: v for k, v in full_env.items() if k != required_key}
        with pytest.raises((ValueError, KeyError)):
            op.build_operator_app(profile_path, partial_env)

    # Noncanonical or malformed Backend ingestion URLs must fail before applySupported, offline.
    for bad_backend_url in (
        "https://backend.example/v1/ingestion",
        "ftp://backend.example",
        "https://user:pass@backend.example",
        "https://backend.example?x=1",
    ):
        bad_url_env = {**full_env, "SMARTSITE_AI_BACKEND_INGESTION_URL": bad_backend_url}
        with pytest.raises(ValueError):
            op.build_operator_app(profile_path, bad_url_env)

    # Bare HTTP(S) origin URL is also canonical without calling Backend at construction.
    origin_app, _origin_driver = op.build_operator_app(
        profile_path, {**full_env, "SMARTSITE_AI_BACKEND_INGESTION_URL": "https://backend.example"}
    )
    assert isinstance(origin_app, FastAPI)

    # Mismatched scope/port/camera across profile & manifests reject fail closed.
    bad_scope_dir = tmp_path / "bad-scope"
    bad_scope_dir.mkdir()
    bad_scope_path, _, _ = _operator_bundle(bad_scope_dir, host_scopes=[[str(uuid4()), "other"]])
    with pytest.raises(ValueError):
        op.build_operator_app(bad_scope_path, full_env)

    bad_port_dir = tmp_path / "bad-port"
    bad_port_dir.mkdir()
    bad_port_path, _, _ = _operator_bundle(bad_port_dir, capture_port=62037)
    with pytest.raises(ValueError):
        op.build_operator_app(bad_port_path, full_env)

    # Stock run_source_control default remains test-only.
    stock_manifest = stock_control.HostManifest.model_validate_json(
        Path(str(doc["hostManifest"])).read_bytes()
    )
    stock_app = stock_control.build_source_control_app(stock_manifest, SecretStr(CONTROL_TOKEN))
    stock_client = TestClient(stock_app)
    hdrs = {"Authorization": f"Bearer {CONTROL_TOKEN}"}
    base = f"/v1/source-control/{doc['siteId']}/{doc['cameraExternalId']}"
    assert stock_client.get(f"{base}/capabilities", headers=hdrs).json()["applySupported"] is False
    assert (
        stock_client.post(
            f"{base}/prepare",
            headers=hdrs,
            json={"commandId": CMD1, "receiptId": str(uuid4()), "expectedRevision": 0},
        ).status_code
        == 404
    )

    # Operator factory builds scoped apply+resume app and bounded child SourceProcessDriver.
    app, driver = op.build_operator_app(profile_path, full_env)
    assert isinstance(app, FastAPI)
    assert isinstance(driver, SourceProcessDriver)
    assert driver.scope == (str(doc["siteId"]), str(doc["cameraExternalId"]))
    assert driver.allow_unowned_start is True
    assert driver.resume_manifest_supported is True
    assert driver.resume_camera_id == str(doc["cameraId"])
    assert driver.stop_file == Path(str(doc["stateDirectory"])).resolve() / "STOP"
    assert driver.generation_file.parent == Path(str(doc["stateDirectory"])).resolve()
    assert driver.command == (
        str(Path(sys.executable)),
        "-m",
        "smartsite_ai.tools.source_operator",
        "--profile",
        str(profile_path.resolve()),
        "--child",
    )
    for forbidden in (
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "PGPASSWORD",
        "PGUSER",
        "SMARTSITE_AI_SOURCE_REGISTRY_KEY",
        "SMARTSITE_AI_SOURCE_CONTROL_TOKEN",
    ):
        assert forbidden not in driver.environment
    assert driver.environment["SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"] == PREVIEW_TOKEN
    assert driver.environment["SMARTSITE_AI_BACKEND_SERVICE_TOKEN"] == BACKEND_TOKEN
    assert (
        driver.environment["SMARTSITE_AI_BACKEND_INGESTION_URL"]
        == (full_env["SMARTSITE_AI_BACKEND_INGESTION_URL"])
    )
    assert driver.process is None
    assert asyncio.run(driver.generation_id(driver.scope)) is None

    client = TestClient(app)
    assert client.get(f"{base}/capabilities").status_code == 401
    assert (
        client.get(
            f"/v1/source-control/{uuid4()}/{doc['cameraExternalId']}/capabilities", headers=hdrs
        ).status_code
        == 404
    )
    caps = client.get(f"{base}/capabilities", headers=hdrs)
    assert caps.status_code == 200
    assert caps.json()["applySupported"] is True
    assert caps.json()["configurationResumeSupported"] is True
    assert caps.json()["sourceRevision"] == 0


def test_build_operator_app_uses_preview_token_for_active_preview_and_control_token_for_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = _op()
    profile_path, doc, environ = _operator_bundle(tmp_path)
    cam = str(doc["cameraExternalId"])
    site_id = str(doc["siteId"])
    seen_preview_auth: list[str] = []

    def fail_capture(_source: str, _live: bool) -> object:
        raise AssertionError("Active running camera should probe HTTP preview, not capture device")

    monkeypatch.setattr(stock_control, "capture_source", fail_capture)
    real_async_client = httpx.AsyncClient

    def preview_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/realtime/configuration-image"
        assert request.url.params["cameraExternalId"] == cam
        auth = request.headers.get("authorization", "")
        seen_preview_auth.append(auth)
        return httpx.Response(
            200,
            json=_valid_image_payload(cam, captured_at=datetime.now(UTC) - timedelta(seconds=0.1)),
        )

    monkeypatch.setattr(
        stock_control.httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(
            transport=httpx.MockTransport(preview_handler), **kwargs
        ),
    )

    # 1. Operator app wires separate PREVIEW token for active preview while route keeps CONTROL.
    app, driver = op.build_operator_app(profile_path, environ)
    driver.process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    driver._current_source = "0"
    client = TestClient(app)
    base = f"/v1/source-control/{site_id}/{cam}"
    probe_body = {"kind": "WEBCAM", "deviceId": "usb0"}

    assert (
        client.post(
            f"{base}/probe",
            headers={"Authorization": f"Bearer {PREVIEW_TOKEN}"},
            json=probe_body,
        ).status_code
        == 401
    )
    probe = client.post(
        f"{base}/probe",
        headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
        json=probe_body,
    )
    assert probe.status_code == 200
    assert seen_preview_auth == [f"Bearer {PREVIEW_TOKEN}"]

    # 2. Direct build_source_control_app defaults to control token unless preview_service_token set.
    seen_preview_auth.clear()
    host = stock_control.HostManifest(
        host_id="edge-op",
        label="Operator Edge",
        scopes=[(site_id, cam)],
        webcams={"usb0": 0},
        active_sources=[
            stock_control.ActiveSource(
                site_id=site_id,
                camera_external_id=cam,
                source=SecretStr("0"),
                preview_origin="http://127.0.0.1:62036",
            )
        ],
    )
    compat_app = stock_control.build_source_control_app(
        host, SecretStr(CONTROL_TOKEN), owned_source=lambda _: "0"
    )
    assert (
        TestClient(compat_app)
        .post(
            f"{base}/probe",
            headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
            json=probe_body,
        )
        .status_code
        == 200
    )
    assert seen_preview_auth == [f"Bearer {CONTROL_TOKEN}"]

    seen_preview_auth.clear()
    split_app = stock_control.build_source_control_app(
        host,
        SecretStr(CONTROL_TOKEN),
        owned_source=lambda _: "0",
        preview_service_token=SecretStr(PREVIEW_TOKEN),
    )
    split_client = TestClient(split_app)
    assert (
        split_client.post(
            f"{base}/probe",
            headers={"Authorization": f"Bearer {PREVIEW_TOKEN}"},
            json=probe_body,
        ).status_code
        == 401
    )
    assert (
        split_client.post(
            f"{base}/probe",
            headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
            json=probe_body,
        ).status_code
        == 200
    )
    assert seen_preview_auth == [f"Bearer {PREVIEW_TOKEN}"]


def _valid_child_generation(
    *,
    worker_generation_id: str = GEN1,
    stream_session_id: str = "33333333-3333-4333-8333-333333333333",
    secret: str = "s" * 43,
    expires_at: str | None = None,
) -> dict[str, str]:
    timestamp = (
        expires_at
        if expires_at is not None
        else (datetime.now(UTC) + timedelta(minutes=10))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return {
        "workerGenerationId": worker_generation_id,
        "streamSessionId": stream_session_id,
        "secret": secret,
        "expiresAt": timestamp,
    }


def _resume_manifest_json(
    camera_id: str,
    *,
    worker_generation_id: str = GEN1,
    stream_session_id: str = "33333333-3333-4333-8333-333333333333",
) -> str:
    return json.dumps(
        {
            "generationBinding": {
                "cameraId": camera_id,
                "workerGenerationId": worker_generation_id,
                "streamSessionId": stream_session_id,
            },
            "configuration": {
                "configurationVersion": 1,
                "viewRevision": 1,
                "bundleSha256": "a" * 64,
            },
        }
    )


@pytest.mark.anyio
async def test_run_operator_child_validates_generation_policy_and_scope_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = _op()
    profile_path, doc, environ = _operator_bundle(tmp_path)
    state_dir = Path(str(doc["stateDirectory"]))
    events: list[str] = []

    def fail_acquire(self: ConfigurationProcessLease) -> None:
        events.append("lease")
        raise AssertionError("Lease must not be acquired before child validation passes")

    async def fail_serve(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("serve")
        raise AssertionError("Capture server must not start before validation")

    async def fail_run_manifest(*_args: object, **_kwargs: object) -> int:
        events.append("runtime")
        raise AssertionError("Inference runtime must not start before validation")

    def fail_model(*_args: object, **_kwargs: object) -> object:
        events.append("model")
        raise AssertionError("Model loader must not run")

    monkeypatch.setattr(ConfigurationProcessLease, "acquire", fail_acquire)
    monkeypatch.setattr(run_configuration_capture, "serve_host", fail_serve)
    monkeypatch.setattr(run_camera_runtime, "run_manifest", fail_run_manifest)
    monkeypatch.setattr(run_camera_runtime, "load_yolo11_detector", fail_model)
    monkeypatch.setattr(model_loading, "load_yolo11_detector", fail_model)
    if hasattr(op, "serve_host"):
        monkeypatch.setattr(op, "serve_host", fail_serve)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", fail_run_manifest)

    base_child_env = {
        "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN": environ["SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"],
        "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0",
        "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(_valid_child_generation()),
    }
    expired_ts = (
        (datetime.now(UTC) - timedelta(minutes=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    bad_capture_envs: list[dict[str, str]] = [
        {k: v for k, v in base_child_env.items() if k != "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"},
        {k: v for k, v in base_child_env.items() if k != "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"},
        {k: v for k, v in base_child_env.items() if k != "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"},
        {**base_child_env, "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": "{bad-json"},
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": (
                json.dumps(_valid_child_generation())[:-1] + ', "secret": "' + ("x" * 43) + '"}'
            ),
        },
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(
                _valid_child_generation(secret="s" * 48)
            ),
        },
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(
                _valid_child_generation(secret="!" * 43)
            ),
        },
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(
                _valid_child_generation(expires_at=expired_ts)
            ),
        },
        {**base_child_env, "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "rtsp://user:pass@10.0.0.2/live"},
        {**base_child_env, "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "1"},
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION": _resume_manifest_json(
                str(uuid4())
            ),
        },
        {
            **base_child_env,
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION": _resume_manifest_json(
                str(doc["cameraId"]), worker_generation_id=GEN2
            ),
        },
    ]

    for bad_env in bad_capture_envs:
        try:
            exit_code = await op.run_operator_child(profile_path, bad_env)
        except (ValueError, KeyError, RuntimeError, OSError, CameraRuntimeError):
            pass
        else:
            assert exit_code != 0

    bad_scope_dir = tmp_path / "bad-child-scope"
    bad_scope_dir.mkdir()
    bad_scope_profile, _, _ = _operator_bundle(bad_scope_dir, host_scopes=[[str(uuid4()), "other"]])
    try:
        exit_code = await op.run_operator_child(bad_scope_profile, base_child_env)
    except (ValueError, KeyError, RuntimeError, OSError, CameraRuntimeError):
        pass
    else:
        assert exit_code != 0

    assert events == []
    assert not (state_dir / "generation.json").exists()


@pytest.mark.anyio
async def test_run_operator_child_capture_mode_reuses_capture_runtime_and_serve_host_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = _op()
    profile_path, doc, environ = _operator_bundle(tmp_path)
    state_dir = Path(str(doc["stateDirectory"]))
    lock_dir = state_dir / "locks"
    seen_generations: list[str | None] = []
    seen_model_calls: list[object] = []

    class _FakeSource:
        def __init__(self, stream_id: str) -> None:
            self.source_id = stream_id
            self.is_connected = False

        async def connect(self) -> None:
            self.is_connected = True

        async def read_frame(self) -> None:
            await asyncio.sleep(0.02)
            return None

        async def close(self) -> None:
            self.is_connected = False

    real_ccr = configuration_runtime.ConfigurationCaptureRuntime

    def spy_ccr(*args: object, **kwargs: object) -> object:
        seen_generations.append(kwargs.get("worker_generation_id"))  # type: ignore[arg-type]
        kwargs.setdefault("source_factory", lambda cfg: _FakeSource(cfg.stream_id))
        return real_ccr(*args, **kwargs)  # type: ignore[arg-type]

    async def fake_serve_host(
        config: object,
        *,
        runtime_factory: object = None,
        stop_requested: asyncio.Event | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert runtime_factory is not None
        runtime = runtime_factory(config)  # type: ignore[operator]
        await runtime.start()
        try:
            assert config.resource_id == UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")  # type: ignore[attr-defined]
            assert config.lock_directory == lock_dir  # type: ignore[attr-defined]
            assert config.token.get_secret_value() == PREVIEW_TOKEN  # type: ignore[attr-defined]
            if stop_requested is not None:
                (state_dir / "STOP").write_text("stop", encoding="utf-8")
                await asyncio.wait_for(stop_requested.wait(), timeout=2.0)
        finally:
            await runtime.stop()
        return {"state": "CONFIGURATION_HOST_STOPPED", "nativeReleaseProved": True}

    async def fail_run_manifest(*args: object, **kwargs: object) -> int:
        seen_model_calls.append((args, kwargs))
        return 99

    def fail_model(*args: object, **kwargs: object) -> object:
        seen_model_calls.append((args, kwargs))
        raise AssertionError("Capture mode must never load a model")

    monkeypatch.setattr(configuration_runtime, "ConfigurationCaptureRuntime", spy_ccr)
    monkeypatch.setattr(run_configuration_capture, "serve_host", fake_serve_host)
    monkeypatch.setattr(run_camera_runtime, "run_manifest", fail_run_manifest)
    monkeypatch.setattr(run_camera_runtime, "load_yolo11_detector", fail_model)
    monkeypatch.setattr(model_loading, "load_yolo11_detector", fail_model)
    if hasattr(op, "ConfigurationCaptureRuntime"):
        monkeypatch.setattr(op, "ConfigurationCaptureRuntime", spy_ccr)
    if hasattr(op, "serve_host"):
        monkeypatch.setattr(op, "serve_host", fake_serve_host)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", fail_run_manifest)

    gen = _valid_child_generation()
    child_env = {
        "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN": environ["SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"],
        "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0",
        "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(gen),
    }
    code = await op.run_operator_child(profile_path, child_env)
    assert code == 0
    assert seen_generations == [GEN1]
    assert seen_model_calls == []
    assert not (state_dir / "generation.json").exists()

    # Shared lease was cleanly released after drain.
    probe_lease = ConfigurationProcessLease(lock_dir, UUID("e98f6475-0660-4c71-a466-4c93d1739ee0"))
    probe_lease.acquire()
    probe_lease.release()


@pytest.mark.anyio
async def test_run_operator_child_inference_owns_lease_args_and_drains_on_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = _op()
    profile_path, doc, environ = _operator_bundle(tmp_path)
    state_dir = Path(str(doc["stateDirectory"]))
    lock_dir = state_dir / "locks"
    resource_id = UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")
    recorded_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    seen_hosts: list[str | None] = []
    cleanup_drained = False

    async def fake_run_manifest(*args: object, **kwargs: object) -> int:
        nonlocal cleanup_drained
        recorded_calls.append((args, kwargs))
        seen_hosts.append(os.environ.get("SMARTSITE_AI_HOST"))
        # Shared physical resource lease must already be owned by the child wrapper.
        contender = ConfigurationProcessLease(lock_dir, resource_id)
        with pytest.raises(RuntimeError):
            contender.acquire()

        (state_dir / "STOP").write_text("stop", encoding="utf-8")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            # Drain cleanup while shared lease remains held until run_manifest exits.
            await asyncio.sleep(0.05)
            contender_during_drain = ConfigurationProcessLease(lock_dir, resource_id)
            with pytest.raises(RuntimeError):
                contender_during_drain.acquire()
            cleanup_drained = True
            raise
        return 0

    monkeypatch.setattr(run_camera_runtime, "run_manifest", fake_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", fake_run_manifest)

    child_env = {
        "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN": environ["SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"],
        "SMARTSITE_AI_BACKEND_SERVICE_TOKEN": environ["SMARTSITE_AI_BACKEND_SERVICE_TOKEN"],
        "SMARTSITE_AI_BACKEND_INGESTION_URL": environ["SMARTSITE_AI_BACKEND_INGESTION_URL"],
        "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0",
        "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(_valid_child_generation()),
        "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION": _resume_manifest_json(
            str(doc["cameraId"])
        ),
    }

    # Contention check: held lease blocks runtime before calling run_manifest.
    blocker = ConfigurationProcessLease(lock_dir, resource_id)
    blocker.acquire()
    try:
        try:
            blocked_code = await op.run_operator_child(profile_path, child_env)
        except (RuntimeError, ValueError, OSError, CameraRuntimeError):
            pass
        else:
            assert blocked_code != 0
        assert recorded_calls == []
    finally:
        blocker.release()

    # Happy resume path with cooperative STOP drain, forced loopback host, and caller env restore.
    monkeypatch.setenv("SMARTSITE_AI_HOST", "0.0.0.0")
    hostile_env = {**child_env, "SMARTSITE_AI_HOST": "0.0.0.0"}
    code = await op.run_operator_child(profile_path, hostile_env)
    assert code == 0
    assert cleanup_drained is True
    assert seen_hosts == ["127.0.0.1"]
    assert os.environ.get("SMARTSITE_AI_HOST") == "0.0.0.0"
    assert len(recorded_calls) == 1
    args, kwargs = recorded_calls[0]
    manifest_arg = args[0] if args else kwargs["manifest_path"]
    assert Path(str(manifest_arg)).resolve() == Path(str(doc["runtimeManifest"])).resolve()
    assert Path(str(kwargs["source_control_profile"])).resolve() == (
        Path(str(doc["hostManifest"])).resolve()
    )
    assert kwargs["expected_camera_id"] == UUID(str(doc["cameraId"]))
    assert kwargs["preview_port"] == 62036
    assert kwargs["require_resume_manifest"] is True
    assert kwargs.get("generation_credential_file") is None
    assert kwargs.get("expected_outbox") is None

    # Lease must be released only after drained stop returns.
    after_lease = ConfigurationProcessLease(lock_dir, resource_id)
    after_lease.acquire()
    after_lease.release()

    # Failures (nonzero return or raised exception) retain shared lease until process exit.
    (state_dir / "STOP").unlink(missing_ok=True)
    real_release = ConfigurationProcessLease.release
    retained_leases: list[ConfigurationProcessLease] = []
    release_calls: list[ConfigurationProcessLease] = []

    def spy_release(self: ConfigurationProcessLease) -> None:
        release_calls.append(self)
        real_release(self)

    real_acquire = ConfigurationProcessLease.acquire

    def spy_acquire(self: ConfigurationProcessLease) -> None:
        real_acquire(self)
        retained_leases.append(self)

    monkeypatch.setattr(ConfigurationProcessLease, "acquire", spy_acquire)
    monkeypatch.setattr(ConfigurationProcessLease, "release", spy_release)

    async def fail_nonzero_run_manifest(*_args: object, **_kwargs: object) -> int:
        return 2

    monkeypatch.setattr(run_camera_runtime, "run_manifest", fail_nonzero_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", fail_nonzero_run_manifest)
    try:
        assert await op.run_operator_child(profile_path, child_env) == 2
        assert release_calls == []
        with pytest.raises(RuntimeError):
            ConfigurationProcessLease(lock_dir, resource_id).acquire()
    finally:
        for held in retained_leases:
            real_release(held)
        retained_leases.clear()

    async def raise_run_manifest(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("inference crashed")

    monkeypatch.setattr(run_camera_runtime, "run_manifest", raise_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", raise_run_manifest)
    try:
        with pytest.raises(RuntimeError, match="inference crashed"):
            await op.run_operator_child(profile_path, child_env)
        assert release_calls == []
        with pytest.raises(RuntimeError):
            ConfigurationProcessLease(lock_dir, resource_id).acquire()
    finally:
        for held in retained_leases:
            real_release(held)


def test_source_operator_cli_main_parent_child_sanitizes_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    op = _op()
    profile_path, _doc, environ = _operator_bundle(tmp_path)

    # Empty CLI fails cleanly without starting capture or server.
    with pytest.raises(SystemExit):
        op.main([])

    # Missing required control token returns nonzero cleanly and never leaks sibling secrets.
    for key in environ:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SMARTSITE_AI_PREVIEW_SERVICE_TOKEN", PREVIEW_TOKEN)
    err_code = op.main(["--profile", str(profile_path), "--port", "62041"])
    captured = capsys.readouterr()
    assert err_code != 0
    assert PREVIEW_TOKEN not in captured.out + captured.err

    # Valid parent env binds 127.0.0.1, uses explicit --port, and stops child on same event loop.
    for key, val in environ.items():
        monkeypatch.setenv(key, val)
    served: list[tuple[object, str, int]] = []
    serve_loops: list[asyncio.AbstractEventLoop] = []
    stop_loops: list[asyncio.AbstractEventLoop] = []
    stopped_scopes: list[object] = []

    real_build_operator_app = op.build_operator_app

    def wrapped_build_operator_app(
        path: Path, env: dict[str, str]
    ) -> tuple[FastAPI, SourceProcessDriver]:
        app, driver = real_build_operator_app(path, env)
        driver.process = SimpleNamespace(returncode=None)  # type: ignore[assignment]

        async def spy_stop(scope: tuple[str, str]) -> None:
            stop_loops.append(asyncio.get_running_loop())
            stopped_scopes.append(scope)
            driver.process = None

        driver.stop = spy_stop  # type: ignore[method-assign]
        return app, driver

    async def fake_serve(self: uvicorn.Server, sockets: object = None) -> None:
        del sockets
        serve_loops.append(asyncio.get_running_loop())
        served.append((self.config.app, str(self.config.host), int(self.config.port)))

    monkeypatch.setattr(op, "build_operator_app", wrapped_build_operator_app)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    parent_code = op.main(["--profile", str(profile_path), "--port", "62041"])
    assert parent_code == 0
    assert len(served) == 1
    assert isinstance(served[0][0], FastAPI)
    assert served[0][1] == "127.0.0.1"
    assert served[0][2] == 62041
    assert len(serve_loops) == 1
    assert len(stop_loops) == 1
    assert stop_loops[0] is serve_loops[0]
    assert len(stopped_scopes) == 1

    # Child CLI delegates cleanly and sanitizes invalid generation env errors.
    monkeypatch.delenv("SMARTSITE_AI_SOURCE_RUNTIME_GENERATION", raising=False)
    child_err = op.main(["--profile", str(profile_path), "--child"])
    child_out = capsys.readouterr()
    assert child_err != 0
    assert PREVIEW_TOKEN not in child_out.out + child_out.err

    delegated: list[Path] = []

    async def fake_child(path: Path, _env: object = None) -> int:
        delegated.append(path)
        return 0

    monkeypatch.setattr(op, "run_operator_child", fake_child)
    assert op.main(["--profile", str(profile_path), "--child"]) == 0
    assert delegated == [profile_path] or delegated == [profile_path.resolve()]


def test_build_operator_app_forwards_windows_user_dirs_and_excludes_secrets_and_pythonpath(
    tmp_path: Path,
) -> None:
    op = _op()
    profile_path, _doc, environ = _operator_bundle(tmp_path)
    user_profile = tmp_path / "user-profile"
    appdata = user_profile / "AppData" / "Roaming"
    local_appdata = user_profile / "AppData" / "Local"
    temp_dir = local_appdata / "Temp"
    for directory in (user_profile, appdata, local_appdata, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    full_env = {
        **environ,
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "USERPROFILE": str(user_profile),
        "APPDATA": str(appdata),
        "LOCALAPPDATA": str(local_appdata),
        "DATABASE_URL": "postgres://secret:db@localhost/smartsite",
        "TEST_DATABASE_URL": "postgres://secret:test@localhost/smartsite_test",
        "PGPASSWORD": "db-password-never-forward",
        "PGUSER": "postgres",
        "PYTHONPATH": str(tmp_path / "injected-pythonpath"),
        "OPENAI_API_KEY": "sk-never-forward",
        "ARBITRARY_SECRET": "never-forward",
    }

    _app, driver = op.build_operator_app(profile_path, full_env)

    assert driver.environment["USERPROFILE"] == str(user_profile)
    assert driver.environment["APPDATA"] == str(appdata)
    assert driver.environment["LOCALAPPDATA"] == str(local_appdata)

    for forbidden in (
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "PGPASSWORD",
        "PGUSER",
        "SMARTSITE_AI_SOURCE_CONTROL_TOKEN",
        "SMARTSITE_AI_SOURCE_REGISTRY_KEY",
        "PYTHONPATH",
        "OPENAI_API_KEY",
        "ARBITRARY_SECRET",
    ):
        assert forbidden not in driver.environment

    allowed = {
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
    assert set(driver.environment) <= allowed


def test_windows_operator_forwarded_env_resolves_home_for_offline_model_import(
    tmp_path: Path,
) -> None:
    op = _op()
    profile_path, _doc, environ = _operator_bundle(tmp_path)
    user_profile = tmp_path / "home-profile"
    appdata = user_profile / "AppData" / "Roaming"
    local_appdata = user_profile / "AppData" / "Local"
    temp_dir = local_appdata / "Temp"
    for directory in (user_profile, appdata, local_appdata, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    parent_env = {
        **environ,
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "USERPROFILE": str(user_profile),
        "APPDATA": str(appdata),
        "LOCALAPPDATA": str(local_appdata),
        "PYTHONPATH": str(tmp_path / "never-forward"),
        "DATABASE_URL": "postgres://secret:db@localhost/smartsite",
    }
    _app, driver = op.build_operator_app(profile_path, parent_env)

    # Confirm the exact stripped-env failure mode on Windows when USERPROFILE is absent.
    if sys.platform == "win32":
        stripped_without_home = {
            k: v
            for k, v in driver.environment.items()
            if k not in {"USERPROFILE", "APPDATA", "LOCALAPPDATA"}
        }
        stripped_probe = subprocess.run(
            [sys.executable, "-c", "from pathlib import Path; Path.home()"],
            env=stripped_without_home,
            cwd=str(temp_dir),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert stripped_probe.returncode != 0
        assert "Could not determine home directory" in stripped_probe.stderr

    # Verify the forwarded child environment resolves Windows home/config paths offline.
    probe = "\n".join(
        (
            "import importlib.util, json, ntpath, os, sys",
            "from pathlib import Path",
            "home = str(Path.home()) if sys.platform == 'win32' else ntpath.expanduser('~')",
            "ultralytics_ok = None",
            "if importlib.util.find_spec('ultralytics') is not None:",
            "    os.environ['YOLO_AUTOINSTALL'] = 'false'",
            "    import ultralytics",
            "    ultralytics_ok = bool(ultralytics.__version__)",
            "print(json.dumps({",
            "    'home': home,",
            "    'appdata': os.environ.get('APPDATA'),",
            "    'localAppdata': os.environ.get('LOCALAPPDATA'),",
            "    'ultralyticsOk': ultralytics_ok,",
            "}))",
        )
    )
    child_probe = subprocess.run(
        [sys.executable, "-c", probe],
        env=driver.environment,
        cwd=str(temp_dir),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child_probe.returncode == 0, child_probe.stderr
    payload = json.loads(child_probe.stdout.strip().splitlines()[-1])
    assert payload["home"] == str(user_profile)
    assert payload["appdata"] == str(appdata)
    assert payload["localAppdata"] == str(local_appdata)
    assert payload["ultralyticsOk"] in (None, True)


@pytest.mark.anyio
async def test_run_operator_child_persists_sanitized_runtime_outcome_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = _op()

    def setup(
        name: str,
    ) -> tuple[Path, dict[str, object], dict[str, str], Path, str]:
        case_dir = tmp_path / name
        case_dir.mkdir(parents=True, exist_ok=True)
        p_path, p_doc, p_env = _operator_bundle(case_dir)
        s_dir = Path(str(p_doc["stateDirectory"])).resolve()
        return p_path, p_doc, p_env, s_dir / "runtime-outcome.json", str(p_doc["cameraExternalId"])

    def build_env(p_doc: dict[str, object], p_env: dict[str, str]) -> dict[str, str]:
        return {
            "SMARTSITE_AI_PREVIEW_SERVICE_TOKEN": p_env["SMARTSITE_AI_PREVIEW_SERVICE_TOKEN"],
            "SMARTSITE_AI_BACKEND_SERVICE_TOKEN": p_env["SMARTSITE_AI_BACKEND_SERVICE_TOKEN"],
            "SMARTSITE_AI_BACKEND_INGESTION_URL": p_env["SMARTSITE_AI_BACKEND_INGESTION_URL"],
            "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0",
            "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(_valid_child_generation()),
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION": _resume_manifest_json(
                str(p_doc["cameraId"])
            ),
        }

    # 1. Clean allowlisted report persistence strips untrusted keys and secrets.
    profile_path, doc, environ, outcome_path, cam = setup("report")

    async def fake_run_manifest(*_args: object, **kwargs: object) -> int:
        callback = kwargs.get("report_callback")
        assert callable(callback)
        callback(
            {
                "token": PREVIEW_TOKEN,
                "secret": BACKEND_TOKEN,
                "imageDataUrl": "data:image/jpeg;base64,SECRETIMG",
                "message": "rtsp://admin:hunter2@10.0.0.1/live",
                "modelLoaded": True,
                "modelClosed": True,
                "cameras": [
                    {
                        "streamId": "cam-stream-1",
                        "cameraExternalId": cam,
                        "status": "failed",
                        "framesProcessed": 5,
                        "eventsEnqueued": 5,
                        "outbox": {
                            "pending": 0,
                            "delivered": 5,
                            "terminal": 0,
                            "secret": "drop-me",
                        },
                        "errorClass": "TrackedOutboxCapacityReached",
                        "message": "rtsp://admin:hunter2@10.0.0.1/live",
                    }
                ],
            }
        )
        return 1

    monkeypatch.setattr(run_camera_runtime, "run_manifest", fake_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", fake_run_manifest)

    code = await op.run_operator_child(profile_path, build_env(doc, environ))
    assert code == 1
    assert outcome_path.is_file()
    raw_outcome = outcome_path.read_text(encoding="utf-8")
    assert json.loads(raw_outcome) == {
        "modelLoaded": True,
        "modelClosed": True,
        "cameras": [
            {
                "streamId": "cam-stream-1",
                "cameraExternalId": cam,
                "status": "failed",
                "framesProcessed": 5,
                "eventsEnqueued": 5,
                "outbox": {"pending": 0, "delivered": 5, "terminal": 0},
                "errorClass": "TrackedOutboxCapacityReached",
            }
        ],
    }
    for forbidden in (PREVIEW_TOKEN, BACKEND_TOKEN, "SECRETIMG", "hunter2", "rtsp://", "drop-me"):
        assert forbidden not in raw_outcome

    # 2. Exception replaces pre-existing stale report with failure CLASS only.
    crash_profile, crash_doc, crash_env, crash_outcome, _ = setup("crash")
    crash_outcome.write_text(json.dumps({"stale": True}), encoding="utf-8")

    async def crash_run_manifest(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError(f"boom token={BACKEND_TOKEN} rtsp://user:pass@10.0.0.9/live")

    monkeypatch.setattr(run_camera_runtime, "run_manifest", crash_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", crash_run_manifest)

    with pytest.raises(RuntimeError):
        await op.run_operator_child(crash_profile, build_env(crash_doc, crash_env))
    crash_raw = crash_outcome.read_text(encoding="utf-8")
    assert json.loads(crash_raw) == {"errorClass": "RuntimeError"}
    for forbidden in (BACKEND_TOKEN, "boom", "rtsp://", "10.0.0.9", "user:pass", "stale"):
        assert forbidden not in crash_raw

    # 3. Unsafe non-identifier errorClass in report payload is stripped cleanly.
    dirty_profile, dirty_doc, dirty_env, dirty_outcome, dirty_cam = setup("dirty")

    async def dirty_error_class_run_manifest(*_args: object, **kwargs: object) -> int:
        callback = kwargs.get("report_callback")
        assert callable(callback)
        callback(
            {
                "modelLoaded": False,
                "modelClosed": False,
                "cameras": [
                    {
                        "streamId": "cam-stream-1",
                        "cameraExternalId": dirty_cam,
                        "status": "failed",
                        "errorClass": "Bad rtsp://u:p@10.0.0.1/live",
                    }
                ],
            }
        )
        return 1

    monkeypatch.setattr(run_camera_runtime, "run_manifest", dirty_error_class_run_manifest)
    if hasattr(op, "run_manifest"):
        monkeypatch.setattr(op, "run_manifest", dirty_error_class_run_manifest)
    assert await op.run_operator_child(dirty_profile, build_env(dirty_doc, dirty_env)) == 1
    sanitized_raw = dirty_outcome.read_text(encoding="utf-8")
    assert "errorClass" not in json.loads(sanitized_raw)["cameras"][0]
    assert "rtsp://" not in sanitized_raw
    assert "u:p@10.0.0.1" not in sanitized_raw

    # 4. Pre-runtime lease acquisition failure records failure CLASS without leaking message.
    lease_profile, lease_doc, lease_env, lease_outcome, _ = setup("lease")

    def fail_acquire(self: ConfigurationProcessLease) -> None:
        raise PermissionError(f"lease denied {PREVIEW_TOKEN}")

    monkeypatch.setattr(ConfigurationProcessLease, "acquire", fail_acquire)
    with pytest.raises(PermissionError):
        await op.run_operator_child(lease_profile, build_env(lease_doc, lease_env))
    lease_raw = lease_outcome.read_text(encoding="utf-8")
    assert json.loads(lease_raw) == {"errorClass": "PermissionError"}
    assert PREVIEW_TOKEN not in lease_raw
    assert "lease denied" not in lease_raw

    # 5. Symlink at outcome target is refused cleanly and never overwrites target contents.
    sym_profile, sym_doc, sym_env, sym_outcome, _ = setup("symlink")
    target = tmp_path / "sensitive-target.txt"
    target.write_text("do-not-touch", encoding="utf-8")
    try:
        sym_outcome.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    with pytest.raises((ValueError, OSError)):
        await op.run_operator_child(sym_profile, build_env(sym_doc, sym_env))
    assert target.read_text(encoding="utf-8") == "do-not-touch"
