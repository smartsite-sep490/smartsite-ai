"""Actual CLI preflight ordering; no model, camera or external API execution."""

import asyncio
import json
import socket
from types import SimpleNamespace

import pytest
from test_configuration_capture_host import configuration

from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.runtime.supervisor import failed
from smartsite_ai.tools import run_camera_runtime as cli


def inputs(tmp_path, monkeypatch, *, port=0):
    config = configuration(tmp_path, rawPreviewEnabled=True, port=port, durationSeconds=2)
    monkeypatch.setenv("SMARTSITE_AI_CAPTURE_SOURCE", "0")
    monkeypatch.setenv("SMARTSITE_AI_CAPTURE_TOKEN", config.token.get_secret_value())
    model = tmp_path / "model.json"
    model.write_text("{}")
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "1",
                "modelSpec": str(model),
                "cameras": [
                    {
                        "streamId": config.stream.stream_id,
                        "cameraId": "11111111-1111-4111-8111-111111111111",
                        "cameraExternalId": config.stream.camera_external_id,
                        "ppeRegionId": "22222222-2222-4222-8222-222222222222",
                        "source": {"kind": "env", "name": "SMARTSITE_AI_CAPTURE_SOURCE"},
                        "live": True,
                        "outbox": str(tmp_path / "outbox.sqlite3"),
                    }
                ],
            }
        )
    )
    return config, path, tmp_path / "host.json"


def test_cli_accepts_repeatable_opt_in_without_changing_default():
    assert cli.build_parser().parse_args(["--manifest", "main.json"]).preview_manifest == []
    args = cli.build_parser().parse_args(
        [
            "--manifest",
            "main.json",
            "--preview-manifest",
            "a.json",
            "--preview-manifest",
            "b.json",
        ]
    )
    assert [p.name for p in args.preview_manifest] == ["a.json", "b.json"]


def test_occupied_preview_port_stops_actual_cli_before_preflight_or_model(tmp_path, monkeypatch):
    touched = []

    def never(*args, **kwargs):
        touched.append(True)
        raise AssertionError("must reject port before model/configuration API")

    monkeypatch.setattr(cli, "preflight_manifest", never)
    monkeypatch.setattr(cli, "load_yolo11_detector", never)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        config, path, preview = inputs(tmp_path, monkeypatch, port=occupied.getsockname()[1])
        with pytest.raises(CameraRuntimeError, match="preview reservation"):
            asyncio.run(
                cli.run_manifest(
                    path, settings=SimpleNamespace(), preview_manifest_paths=(preview,)
                )
            )
    assert touched == []
    lease = ConfigurationProcessLease(config.lock_directory, config.resource_id)
    lease.acquire()
    lease.release()


@pytest.mark.parametrize("stage", ["preflight", "model"])
def test_cli_failure_releases_all_reserved_ports_and_leases(tmp_path, monkeypatch, capsys, stage):
    config, path, preview = inputs(tmp_path, monkeypatch)
    models = []

    async def preflight(*args, **kwargs):
        if stage == "preflight":
            return (
                failed(config.stream.stream_id, config.stream.camera_external_id, "Synthetic"),
            ), ()
        return None, ()

    def model(*args, **kwargs):
        models.append(True)
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(cli, "preflight_manifest", preflight)
    monkeypatch.setattr(cli, "load_yolo11_detector", model)
    if stage == "model":
        with pytest.raises(RuntimeError, match="synthetic model failure"):
            asyncio.run(
                cli.run_manifest(
                    path, settings=SimpleNamespace(), preview_manifest_paths=(preview,)
                )
            )
    else:
        assert (
            asyncio.run(
                cli.run_manifest(
                    path, settings=SimpleNamespace(), preview_manifest_paths=(preview,)
                )
            )
            == 1
        )
    assert models == ([True] if stage == "model" else [])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    binding = lines[0]
    assert binding["state"] == "RAW_PREVIEW_BOUND_NOT_READY"
    assert config.token.get_secret_value() not in json.dumps(lines)
    lease = ConfigurationProcessLease(config.lock_directory, config.resource_id)
    lease.acquire()
    lease.release()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", binding["port"]))


def test_actual_cli_assembly_shares_capture_and_closes_model_after_source(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import httpx
    from configuration_capture_fakes import FakeSource, frame, until
    from test_multi_camera_runtime import REGION_A, _configuration
    from test_processing_preview_host import Server
    from test_processing_worker import FakeDetector

    from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
    from smartsite_ai.integrations.backend_client import BackendClient
    from smartsite_ai.runtime.assembly import CameraPlan
    from smartsite_ai.runtime.preview_host import PreviewHostGroup

    config, path, preview = inputs(tmp_path, monkeypatch)
    source, servers, closed, groups = FakeSource(), [], [], []
    source.source_id = config.stream.stream_id

    class ConfigurationClient:
        async def aclose(self):
            closed.append("configuration")

    class Poller:
        def ensure_fresh(self):
            pass

        async def run(self, stop):
            await stop.wait()

    async def preflight(bound, **kwargs):
        store = RegionConfigurationStore()
        await store.apply(_configuration(config.stream.camera_external_id, REGION_A))
        return None, (CameraPlan(bound.cameras[0], store, Poller(), ConfigurationClient()),)

    runner = cli.UltralyticsYoloRunner()

    def close_model():
        assert source.closes == 1
        closed.append("model")

    runner.close = close_model

    def server_factory(app):
        server = Server(app)
        servers.append(server)
        return server

    def hosts_factory(*args):
        hosts = PreviewHostGroup(*args, server_factory=server_factory)
        groups.append(hosts)
        return hosts

    def no_backend_requests(request):
        raise AssertionError("synthetic detector has no finding to send")

    monkeypatch.setattr(cli, "preflight_manifest", preflight)
    monkeypatch.setattr(
        cli, "load_yolo11_detector", lambda *a, **k: (FakeDetector(), runner, None, None)
    )
    monkeypatch.setattr(cli, "OpenCvFrameSource", lambda _: source)
    monkeypatch.setattr(
        cli,
        "PreviewHostGroup",
        hosts_factory,
    )
    monkeypatch.setattr(
        cli.BackendClient,
        "from_settings",
        lambda _: BackendClient(
            "http://backend:3000", "synthetic", transport=httpx.MockTransport(no_backend_requests)
        ),
    )

    async def scenario():
        seen = asyncio.Event()
        task = asyncio.create_task(
            cli.run_manifest(path, settings=SimpleNamespace(), preview_manifest_paths=(preview,))
        )
        viewer = None
        try:
            await until(lambda: len(servers) == 1 and source.connects == 1)
            source.items.put_nowait(
                frame(
                    stream_id=config.stream.stream_id,
                    camera_external_id=config.stream.camera_external_id,
                    captured_at=datetime.now(UTC),
                )
            )
            await until(
                lambda: groups[0].observer_for(config.stream.stream_id).latest() is not None
            )

            async def observed_app(scope, receive, send):
                async def observed_send(message):
                    await send(message)
                    if (
                        scope["path"] == "/v1/realtime/video-stream"
                        and message["type"] == "http.response.body"
                        and message.get("body")
                    ):
                        assert json.loads(message["body"])["sequenceNumber"] == "0"
                        seen.set()

                await servers[0].app(scope, receive, observed_send)

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(observed_app), base_url="http://test"
            ) as client:
                viewer = asyncio.create_task(
                    client.get(
                        "/v1/realtime/video-stream",
                        params={"cameraExternalId": config.stream.camera_external_id},
                        headers={"Authorization": f"Bearer {config.token.get_secret_value()}"},
                    )
                )
                await asyncio.wait_for(seen.wait(), 2)
                await until(
                    lambda: (
                        groups[0].observer_for(config.stream.stream_id).configuration.latest()
                        is not None
                    )
                )
                snapshot = await client.get(
                    "/v1/realtime/configuration-image",
                    params={"cameraExternalId": config.stream.camera_external_id},
                    headers={"Authorization": f"Bearer {config.token.get_secret_value()}"},
                )
                assert snapshot.status_code == 200
                assert snapshot.json()["purpose"] == "REGION_CONFIGURATION"
                assert snapshot.json()["frame"]["sequenceNumber"] == "0"
                viewer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await viewer
                assert source.connects == 1 and source.closes == 0
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert closed == ["model", "configuration"]
            competitor = ConfigurationProcessLease(config.lock_directory, config.resource_id)
            competitor.acquire()
            competitor.release()
        finally:
            if viewer is not None and not viewer.done():
                viewer.cancel()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
