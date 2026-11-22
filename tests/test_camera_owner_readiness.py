"""Owned camera readiness RED/GREEN tests reusing existing synthetic capture/preview fakes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from configuration_capture_fakes import NOW, FakeSource, finish, frame, until
from fastapi import FastAPI
from pydantic import SecretStr
from test_camera_media_observer import until as until_preview
from test_configuration_capture import capture_pair
from test_multi_camera_runtime import _clean_result, _StopPoller
from test_source_control_child_profile import inputs as child_profile_inputs

from smartsite_ai.camera_media_api import camera_media_router
from smartsite_ai.config import Settings
from smartsite_ai.configuration_image_api import configuration_image_router
from smartsite_ai.configuration_runtime import ConfigurationCaptureRuntime
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import PreviewFrameQueue
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.testing import FakeFrameSource
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.runtime import source_control_child as child_mod
from smartsite_ai.runtime.analysis_preview import AnalysisPreviewObserver
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.runtime.media_host import run_with_camera_preview
from smartsite_ai.runtime.media_preview import CameraPreviewObserver
from smartsite_ai.runtime.supervisor import (
    CameraOutcome,
    CameraSession,
    RuntimePlan,
    RuntimeReport,
)
from smartsite_ai.tools import run_camera_runtime as runtime_cli

TOKEN = "synthetic-readiness-token-32-chars-minimum"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
HEADER = "X-SmartSite-Worker-Generation"
PATH = "/v1/realtime/configuration-image"
RESOURCE = uuid4()
GEN1 = "11111111-2222-4333-8444-555555555555"
GEN2 = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def _preview_frame(stream_id: str, camera_external_id: str, captured_at: datetime) -> FrameEnvelope:
    return FrameEnvelope.model_validate(
        {
            "stream_id": stream_id,
            "camera_external_id": camera_external_id,
            "session_id": uuid4(),
            "sequence_number": 0,
            "captured_at": captured_at,
            "width": 64,
            "height": 64,
            "payload": b"\x00\x20\x40" * 64 * 64,
        }
    )


class _FakeAnalysis:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self._payload = payload

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def latest(self) -> dict[str, object] | None:
        return self._payload


def _capture_app(capture: object, **kwargs: object) -> FastAPI:
    app = FastAPI()
    app.include_router(
        configuration_image_router(capture, SecretStr(TOKEN), **kwargs)  # type: ignore[arg-type]
    )
    return app


def test_configuration_image_router_validates_and_emits_header_only_on_fresh_success() -> None:
    capture_idle, _, _, _ = capture_pair()
    for invalid in ("", "   ", "not-a-uuid", 123):
        with pytest.raises((ValueError, TypeError)):
            configuration_image_router(
                capture_idle,
                SecretStr(TOKEN),
                worker_generation_id=invalid,  # type: ignore[arg-type]
            )

    async def scenario() -> None:
        now = [NOW]
        source = FakeSource()
        capture, _, _, _ = capture_pair(source, kind="LIVE", clock=lambda: now[0])
        default_app = _capture_app(capture)
        bound_app = _capture_app(capture, worker_generation_id=GEN1.upper())

        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=default_app), base_url="http://test"
            ) as default_client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=bound_app), base_url="http://test"
            ) as bound_client,
        ):
            before_start = await bound_client.get(
                PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
            )
            assert before_start.status_code == 503
            assert HEADER not in before_start.headers

            await capture.start()
            try:
                await source.reading.wait()
                source.items.put_nowait(frame())
                await until(lambda: capture.latest() is not None)
                expected_body = capture.latest()
                assert expected_body is not None

                default_ok = await default_client.get(
                    PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
                )
                assert default_ok.status_code == 200
                assert default_ok.json() == expected_body
                assert HEADER not in default_ok.headers

                bound_ok = await bound_client.get(
                    PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
                )
                assert bound_ok.status_code == 200
                assert bound_ok.json() == expected_body
                assert bound_ok.headers[HEADER] == GEN1
                assert bound_ok.headers["Cache-Control"] == "no-store"
                assert bound_ok.headers["X-Content-Type-Options"] == "nosniff"

                for query, hdrs, status in [
                    ("cameraExternalId=camera-a", {}, 401),
                    ("cameraExternalId=camera-a", {"Authorization": "Bearer wrong"}, 401),
                    ("cameraExternalId=", AUTH, 400),
                    ("cameraExternalId=camera-a&source=extra", AUTH, 400),
                    ("cameraExternalId=other-camera", AUTH, 404),
                ]:
                    err = await bound_client.get(f"{PATH}?{query}", headers=hdrs)
                    assert err.status_code == status
                    assert HEADER not in err.headers

                # Stale capture returns 503 and must drop the generation header.
                now[0] = NOW + timedelta(seconds=6)
                stale = await bound_client.get(
                    PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
                )
                assert stale.status_code == 503
                assert HEADER not in stale.headers
            finally:
                await finish(capture, source)

            after_stop = await bound_client.get(
                PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
            )
            assert after_stop.status_code == 503
            assert HEADER not in after_stop.headers

    asyncio.run(scenario())


def test_configuration_capture_runtime_forwards_generation_without_starting_source_or_model(
    tmp_path: Path,
) -> None:
    stream = StreamConfig(
        stream_id="configuration-stream",
        camera_external_id="camera-a",
        source_url="0",
        is_live=True,
    )
    forbidden_builds: list[StreamConfig] = []

    def fail_factory(cfg: StreamConfig) -> FakeSource:
        forbidden_builds.append(cfg)
        raise AssertionError("source_factory must not run before start()")

    with pytest.raises((ValueError, TypeError)):
        ConfigurationCaptureRuntime(
            stream=stream,
            source_kind="LIVE",
            resource_id=RESOURCE,
            lock_directory=tmp_path,
            service_token=SecretStr(TOKEN),
            source_factory=fail_factory,
            worker_generation_id="invalid-uuid",  # type: ignore[call-arg]
        )
    assert forbidden_builds == []

    async def scenario() -> None:
        source = FakeSource()
        builds: list[StreamConfig] = []
        before_modules = set(sys.modules)

        def build(cfg: StreamConfig) -> FakeSource:
            builds.append(cfg)
            return source

        lock_dir = tmp_path / "bound"
        lock_dir.mkdir()
        runtime = ConfigurationCaptureRuntime(
            stream=stream,
            source_kind="LIVE",
            resource_id=RESOURCE,
            lock_directory=lock_dir,
            service_token=SecretStr(TOKEN),
            source_factory=build,
            clock=lambda: NOW,
            stop_timeout=0.05,
            worker_generation_id=GEN1.upper(),  # type: ignore[call-arg]
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime.app), base_url="http://test"
        ) as client:
            idle = await client.get(PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH)
            assert idle.status_code == 503
            assert HEADER not in idle.headers
            assert builds == [] and source.connects == source.reads == 0

            await runtime.start()
            try:
                await source.reading.wait()
                source.items.put_nowait(frame())
                ok: httpx.Response | None = None
                async with asyncio.timeout(2):
                    while True:
                        res = await client.get(
                            PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH
                        )
                        if res.status_code == 200:
                            ok = res
                            break
                        await asyncio.sleep(0.001)
                assert ok is not None
                assert ok.headers[HEADER] == GEN1
                assert ok.json()["sourceKind"] == "LIVE"
                assert "workerGenerationId" not in ok.json()
            finally:
                await finish(runtime, source)

            stopped = await client.get(PATH, params={"cameraExternalId": "camera-a"}, headers=AUTH)
            assert stopped.status_code == 503
            assert HEADER not in stopped.headers

        assert not ({"torch", "ultralytics"} & (set(sys.modules) - before_modules))

    asyncio.run(scenario())


@pytest.mark.anyio
async def test_camera_media_router_freezes_validates_and_scopes_generation_header() -> None:
    empty_observers: dict[str, CameraPreviewObserver] = {}
    for bad_map in (
        {"": GEN1},
        {"   ": GEN1},
        {"a" * 257: GEN1},
        {"ext-a": ""},
        {"ext-a": "not-a-uuid"},
        {123: GEN1},
    ):
        with pytest.raises((ValueError, TypeError)):
            camera_media_router(
                empty_observers,
                SecretStr(TOKEN),
                worker_generation_ids=bad_map,  # type: ignore[arg-type]
            )

    # Default omitted worker_generation_ids never attaches generation header.
    default_router = camera_media_router(empty_observers, SecretStr(TOKEN))
    assert default_router is not None

    # Supply bindings BEFORE observers exist; caller mutation afterwards must not affect router.
    mutable_ids: dict[str, str] = {"ext-a": GEN1.upper()}
    app = FastAPI()
    app.include_router(
        camera_media_router(
            empty_observers,
            SecretStr(TOKEN),
            max_stream_seconds=1.0,
            worker_generation_ids=mutable_ids,  # type: ignore[call-arg]
        )
    )
    mutable_ids["ext-a"] = GEN2
    mutable_ids["ext-b"] = GEN2

    now = [datetime.now(UTC)]
    queue_a, queue_b = PreviewFrameQueue(), PreviewFrameQueue()
    streaming = True
    analysis_a = _FakeAnalysis({"detections": []})
    obs_a = CameraPreviewObserver(
        "ext-a",
        queue_a,
        source_kind="LIVE",
        is_streaming=lambda: streaming,
        clock=lambda: now[0],
        analysis=cast(AnalysisPreviewObserver, analysis_a),
    )
    obs_b = CameraPreviewObserver(
        "ext-b",
        queue_b,
        source_kind="LIVE",
        is_streaming=lambda: True,
        clock=lambda: now[0],
    )
    empty_observers["ext-a"] = obs_a
    empty_observers["ext-b"] = obs_b

    await obs_a.start()
    await obs_b.start()
    try:
        frame_a = _preview_frame("stream-a", "ext-a", now[0])
        frame_b = _preview_frame("stream-b", "ext-b", now[0])
        queue_a.put(frame_a)
        queue_b.put(frame_b)
        await until_preview(lambda: obs_a.latest() is not None and obs_b.latest() is not None)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=2
        ) as client:
            res_a = await client.get(f"{PATH}?cameraExternalId=ext-a", headers=AUTH)
            assert res_a.status_code == 200
            assert set(res_a.json()) == {"schemaVersion", "purpose", "sourceKind", "frame"}
            assert res_a.headers[HEADER] == GEN1

            res_b = await client.get(f"{PATH}?cameraExternalId=ext-b", headers=AUTH)
            assert res_b.status_code == 200
            assert set(res_b.json()) == {"schemaVersion", "purpose", "sourceKind", "frame"}
            assert HEADER not in res_b.headers

            for req, hdrs, code in [
                (f"{PATH}?cameraExternalId=ext-a", {}, 401),
                (f"{PATH}?cameraExternalId=", AUTH, 400),
                (f"{PATH}?cameraExternalId=unknown-cam", AUTH, 404),
            ]:
                err = await client.get(req, headers=hdrs)
                assert err.status_code == code
                assert HEADER not in err.headers

            preview_ok = await client.get(
                "/v1/realtime/preview/latest?cameraExternalId=ext-a", headers=AUTH
            )
            assert preview_ok.status_code == 200
            assert preview_ok.json() == {"sourceKind": "LIVE", "frame": {"detections": []}}
            assert HEADER not in preview_ok.headers

            async with client.stream(
                "GET", "/v1/realtime/video-stream?cameraExternalId=ext-a", headers=AUTH
            ) as stream_res:
                assert stream_res.status_code == 200
                assert HEADER not in stream_res.headers
                line = json.loads(await stream_res.aiter_lines().__anext__())
                assert line["cameraExternalId"] == frame_a.camera_external_id
                assert line["sequenceNumber"] == "0"

            now[0] += timedelta(seconds=5)
            stale = await client.get(f"{PATH}?cameraExternalId=ext-a", headers=AUTH)
            assert stale.status_code == 503
            assert HEADER not in stale.headers
    finally:
        await obs_a.stop()
        await obs_b.stop()


@pytest.mark.anyio
async def test_run_with_camera_preview_forwards_generation_bindings_after_validation() -> None:
    async def forbidden() -> object:
        raise AssertionError("invalid generation bindings must fail before preflight/model")

    with pytest.raises((ValueError, TypeError)):
        await run_with_camera_preview(
            RuntimePlan(
                cast(Callable[[], Awaitable[None]], forbidden),
                cast(Callable[[], Awaitable[DetectorProtocol]], forbidden),
                cast(Callable[[], Awaitable[None]], forbidden),
                lambda _d: (),
            ),
            {},
            SecretStr(TOKEN),
            host="127.0.0.1",
            port=0,
            worker_generation_ids={"ext-media": "not-a-uuid"},  # type: ignore[call-arg]
        )

    frame_m = _preview_frame("media", "ext-media", datetime.now(UTC))
    source = FakeFrameSource("media", [frame_m], block_when_exhausted=True)
    queue = PreviewFrameQueue()
    stream = StreamWorker(
        StreamConfig(
            stream_id="media", camera_external_id="ext-media", source_url="0", is_live=True
        ),
        source=source,
        preview_frames=queue,
    )
    observer = CameraPreviewObserver(
        "ext-media",
        queue,
        source_kind="LIVE",
        is_streaming=lambda: stream.state == StreamState.STREAMING,
    )
    observers: dict[str, CameraPreviewObserver] = {}
    stop, bound = asyncio.Event(), asyncio.Event()
    ports: list[int] = []

    class Worker:
        async def run(self):
            await stream.start()
            try:
                assert await stream.get_frame() is frame_m
                await stop.wait()
                return replace(_clean_result(), frames_processed=1)
            finally:
                await stream.stop()

    async def preflight() -> None:
        return None

    async def open_model() -> DetectorProtocol:
        return cast(DetectorProtocol, object())

    async def close_model() -> None:
        return None

    def sessions(_detector: DetectorProtocol) -> tuple[CameraSession, ...]:
        observers["ext-media"] = observer
        return (CameraSession("media", "ext-media", Worker(), _StopPoller(), preview=observer),)

    def on_bound(port: int) -> None:
        ports.append(port)
        bound.set()

    task = asyncio.create_task(
        run_with_camera_preview(
            RuntimePlan(preflight, open_model, close_model, sessions),
            observers,
            SecretStr(TOKEN),
            host="127.0.0.1",
            port=0,
            on_bound=on_bound,
            worker_generation_ids={"ext-media": GEN1.upper()},  # type: ignore[call-arg]
        )
    )
    try:
        await asyncio.wait_for(bound.wait(), 2)
        await until_preview(lambda: observer.latest() is not None)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{ports[0]}", trust_env=False, timeout=2
        ) as client:
            cfg = await client.get(f"{PATH}?cameraExternalId=ext-media", headers=AUTH)
            assert cfg.status_code == 200
            assert cfg.headers[HEADER] == GEN1
            assert cfg.json()["frame"]["cameraExternalId"] == "ext-media"
            async with client.stream(
                "GET", "/v1/realtime/video-stream?cameraExternalId=ext-media", headers=AUTH
            ) as ndjson:
                assert ndjson.status_code == 200
                assert HEADER not in ndjson.headers
        stop.set()
        report = await task
        assert report.model_closed
    finally:
        stop.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_run_camera_runtime_wires_only_validated_profile_and_rejects_bad_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, bundle, cred_path, original_cred, expected, env = child_profile_inputs(tmp_path)
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    profile_path = tmp_path / "source-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "host_id": "edge",
                "label": "Edge",
                "scopes": [
                    [value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id]
                ],
                "webcams": {"usb0": 0},
            }
        ),
        encoding="utf-8",
    )

    def response(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=bundle.model_dump(mode="json", by_alias=True),
            headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
        )

    monkeypatch.setattr(
        BackendConfigurationClient,
        "from_settings",
        classmethod(
            lambda _cls, *_a, **_kw: BackendConfigurationClient(
                "https://backend.example",
                "synthetic-service",
                policy_v2=True,
                transport=httpx.MockTransport(response),
            )
        ),
    )

    model_calls: list[str] = []

    def fail_if_model_opened(*_args: object, **_kwargs: object) -> object:
        model_calls.append("opened")
        raise AssertionError("model must not open on test or invalid binding")

    monkeypatch.setattr(runtime_cli, "load_yolo11_detector", fail_if_model_opened)

    recorded_preview_calls: list[dict[str, object]] = []

    async def fake_run_with_camera_preview(
        plan: RuntimePlan,
        observers: dict[str, CameraPreviewObserver],
        service_token: SecretStr,
        *,
        host: str,
        port: int,
        on_bound: Callable[[int], None] | None = None,
        worker_generation_ids: dict[str, str] | None = None,
    ) -> RuntimeReport:
        assert isinstance(observers, dict)
        recorded_preview_calls.append(
            {"host": host, "port": port, "worker_generation_ids": worker_generation_ids}
        )
        return RuntimeReport(
            cameras=(
                CameraOutcome(
                    stream_id="synthetic",
                    camera_external_id=bundle.camera_external_id,
                    status="completed",
                    outbox=_clean_result().outbox,
                ),
            ),
            model_loaded=True,
            model_closed=True,
        )

    monkeypatch.setattr(runtime_cli, "run_with_camera_preview", fake_run_with_camera_preview)

    settings = Settings(_env_file=None, preview_service_token=SecretStr(TOKEN))
    outbox_path = Path(value["cameras"][0]["outbox"])
    expected_binding = {bundle.camera_external_id: expected.generationBinding.workerGenerationId}

    # 1. Arbitrary SMARTSITE_AI_SOURCE_RUNTIME_GENERATION without profile rejects early.
    monkeypatch.setenv(
        "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION", env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"]
    )
    with pytest.raises(CameraRuntimeError, match="source-control profile"):
        await runtime_cli.run_manifest(
            manifest_path,
            settings=settings,
            generation_credential_file=cred_path,
            expected_camera_id=UUID(bundle.camera_id),
            expected_outbox=outbox_path,
            preview_port=8123,
        )
    assert recorded_preview_calls == [] and model_calls == []
    monkeypatch.delenv("SMARTSITE_AI_SOURCE_RUNTIME_GENERATION", raising=False)

    # 2. Normal file-bound startup leaves worker_generation_ids=None (no override emitted).
    assert (
        await runtime_cli.run_manifest(
            manifest_path,
            settings=settings,
            generation_credential_file=cred_path,
            expected_camera_id=UUID(bundle.camera_id),
            expected_outbox=outbox_path,
            preview_port=8123,
        )
        == 0
    )
    assert recorded_preview_calls[-1]["worker_generation_ids"] is None

    # 3. Invalid child binding (generation mismatch / missing manifest) rejects early.
    mismatched_gen = json.loads(env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"])
    mismatched_gen["workerGenerationId"] = GEN2
    for mode in ("nt", "posix"):
        with monkeypatch.context() as mp:
            if mode == "posix":
                mp.setattr(runtime_cli, "os", SimpleNamespace(name="posix", environ=os.environ))
                mp.setattr(child_mod, "_persist_credential", lambda *_a: None)
            mp.setenv(
                "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE", env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"]
            )
            mp.setenv("SMARTSITE_AI_SOURCE_RUNTIME_GENERATION", json.dumps(mismatched_gen))
            mp.setenv(
                "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json()
            )
            before_calls = len(recorded_preview_calls)
            with pytest.raises((CameraRuntimeError, ValueError)):
                await runtime_cli.run_manifest(
                    manifest_path,
                    settings=settings,
                    expected_camera_id=UUID(bundle.camera_id),
                    preview_port=8123,
                    source_control_profile=profile_path,
                )
            assert len(recorded_preview_calls) == before_calls and model_calls == []
            mp.delenv("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", raising=False)
            with pytest.raises((CameraRuntimeError, ValueError)):
                await runtime_cli.run_manifest(
                    manifest_path,
                    settings=settings,
                    expected_camera_id=UUID(bundle.camera_id),
                    preview_port=8123,
                    source_control_profile=profile_path,
                )
            assert len(recorded_preview_calls) == before_calls and model_calls == []

    # 4. POSIX profile path: bind_source_control_child validates equality and forwards ID.
    persisted: list[tuple[Path, str]] = []
    with monkeypatch.context() as mp:
        mp.setattr(runtime_cli, "os", SimpleNamespace(name="posix", environ=os.environ))
        mp.setattr(
            child_mod,
            "_persist_credential",
            lambda p, cred: persisted.append((p, cred.worker_generation_id)),
        )
        for key, val in env.items():
            mp.setenv(key, val)
        mp.setenv("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json())
        assert (
            await runtime_cli.run_manifest(
                manifest_path,
                settings=settings,
                expected_camera_id=UUID(bundle.camera_id),
                preview_port=8123,
                source_control_profile=profile_path,
            )
            == 0
        )
    assert persisted == [(cred_path, expected.generationBinding.workerGenerationId)]
    assert recorded_preview_calls[-1]["worker_generation_ids"] == expected_binding
    assert cred_path.read_bytes() == original_cred

    # 5. Windows memory path: bind_source_control_child_memory validates and forwards header ID.
    if os.name == "nt":
        for key, val in env.items():
            monkeypatch.setenv(key, val)
        monkeypatch.setenv(
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json()
        )
        assert (
            await runtime_cli.run_manifest(
                manifest_path,
                settings=settings,
                expected_camera_id=UUID(bundle.camera_id),
                preview_port=8123,
                source_control_profile=profile_path,
            )
            == 0
        )
        assert recorded_preview_calls[-1]["worker_generation_ids"] == expected_binding
        assert cred_path.read_bytes() == original_cred
