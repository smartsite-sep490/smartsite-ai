import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_generation_runtime_claim import CAMERA, REGION, ConfigurationClient, credential_file

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.integrations.outbox import OutboxCounts
from smartsite_ai.processing_worker import ProcessingWorkerResult
from smartsite_ai.runtime.assembly import (
    assemble_sessions,
    close_configuration_clients,
    preflight_manifest,
)
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest
from smartsite_ai.tools import run_camera_worker as cli


@pytest.mark.parametrize("enabled", [False, True])
def test_manifest_selection_reaches_real_worker_and_hash_publisher(tmp_path: Path, enabled):
    async def scenario():
        credential, _ = credential_file(tmp_path)
        model, video = tmp_path / "model.json", tmp_path / "synthetic.mp4"
        model.write_text("{}")
        video.write_bytes(b"synthetic-no-video")
        entry = {
            "streamId": "synthetic",
            "cameraId": str(CAMERA),
            "cameraExternalId": "CAM",
            "ppeRegionId": str(REGION),
            "source": {"kind": "path", "path": str(video)},
            "outbox": str(tmp_path / "outbox.sqlite3"),
            "evidenceDir": str(tmp_path),
            "generationCredentialFile": str(credential),
        }
        if enabled:
            entry["partitionedFrameDelivery"] = True
        manifest = parse_manifest(
            {"schemaVersion": "1", "modelSpec": str(model), "cameras": [entry]}
        )
        assert manifest.cameras[0].partitioned_frame_delivery is enabled
        for value in ["true", 1, None]:
            with pytest.raises(CameraRuntimeError):
                parse_manifest(
                    {
                        "schemaVersion": "1",
                        "modelSpec": str(model),
                        "cameras": [{**entry, "partitionedFrameDelivery": value}],
                    }
                )
        failed, plans = await preflight_manifest(
            bind_manifest(manifest, {}), configuration_client_factory=ConfigurationClient
        )
        assert failed is None
        try:
            source = SimpleNamespace(source_id="synthetic", is_connected=False)
            sessions = assemble_sessions(
                plans,
                SimpleNamespace(),
                backend_client_factory=lambda: SimpleNamespace(),
                source_factory=lambda config: source,
                evidence_max_bytes=1024,
                delivery_interval_seconds=0.25,
            )
            worker = sessions[0].worker
            assert worker._partitioned_frame_delivery is enabled
            assert worker._evidence_publisher._bind_content_hash is enabled
            assert not source.is_connected
        finally:
            await close_configuration_clients(plans)

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [False, True])
def test_single_cli_optin_keeps_default_off_and_reaches_worker_before_any_source_open(
    tmp_path, monkeypatch, enabled
):
    async def scenario():
        credential, _ = credential_file(tmp_path)
        model = tmp_path / "model.json"
        model.write_text("{}")
        args = [
            "--source",
            "0",
            "--stream-id",
            "synthetic",
            "--camera-id",
            str(CAMERA),
            "--camera-external-id",
            "CAM",
            "--ppe-region-id",
            str(REGION),
            "--model-spec",
            str(model),
            "--outbox",
            str(tmp_path / "outbox.sqlite3"),
            "--evidence-dir",
            str(tmp_path),
            "--generation-credential-file",
            str(credential),
        ]
        if enabled:
            args.append("--partitioned-frame-delivery")
        parsed = cli.build_parser().parse_args(args)
        assert parsed.partitioned_frame_delivery is enabled
        monkeypatch.setattr(
            cli.BackendConfigurationClient, "from_settings", lambda settings: ConfigurationClient()
        )
        monkeypatch.setattr(
            cli.BackendClient, "from_settings", lambda settings: ConfigurationClient()
        )
        runner = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(
            cli, "load_yolo11_detector", lambda *a, **k: (SimpleNamespace(), runner, None, None)
        )
        source = SimpleNamespace(source_id="synthetic", is_connected=False)
        monkeypatch.setattr(cli, "OpenCvFrameSource", lambda config: source)

        async def inspect(worker, poller):
            assert worker._partitioned_frame_delivery is enabled
            assert worker._evidence_publisher._bind_content_hash is enabled
            assert not source.is_connected
            assert poller._bound_source is worker._stream.source
            return ProcessingWorkerResult(1, 0, OutboxCounts(0, 0, 0))

        monkeypatch.setattr(cli, "_coordinate", inspect)
        assert await cli.run_worker(parsed, settings=SimpleNamespace()) == 0

    asyncio.run(scenario())


def test_fullbundle_entry_preflight_does_not_require_arbitrary_legacy_anchor():
    async def scenario():
        from smartsite_ai.runtime.assembly import _require_ppe_region

        store = RegionConfigurationStore()
        content = {
            "schemaVersion": "1.0.0",
            "cameraId": str(CAMERA),
            "cameraExternalId": "CAM",
            "configurationVersion": 1,
            "viewRevision": 1,
            "issuedAt": "2026-10-30T00:00:00.000Z",
            "regions": [],
        }
        await store.apply_bundle(
            CameraConfigurationBundle.from_wire_bytes(
                json.dumps(
                    {
                        **content,
                        "bundleSha256": compute_canonical_payload_hash(content),
                    }
                ).encode()
            )
        )
        await _require_ppe_region(store, "CAM", REGION)
        await cli._require_ppe_region(store, "CAM", REGION)
        legacy = RegionConfigurationStore()
        await legacy.apply((await ConfigurationClient().fetch()).configuration)
        from uuid import uuid4

        with pytest.raises(CameraRuntimeError):
            await _require_ppe_region(legacy, "CAM", uuid4())
        with pytest.raises(cli.CameraWorkerRunError):
            await cli._require_ppe_region(legacy, "CAM", uuid4())

    asyncio.run(scenario())
