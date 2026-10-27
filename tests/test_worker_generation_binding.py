import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope

CAMERA = UUID("11111111-1111-4111-8111-111111111111")
NOW = datetime(2026, 10, 27, tzinfo=UTC)


def credential_payload():
    return {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA),
        "workerGenerationId": str(uuid4()),
        "streamSessionId": str(uuid4()),
        "secret": "x" * 43,
        "expiresAt": "2026-10-27T01:00:00.000Z",
    }


def test_credential_load_is_bounded_scoped_expiring_and_redacted(tmp_path: Path):
    from smartsite_ai.integrations.worker_generation import load_worker_generation

    payload = credential_payload()
    path = tmp_path / "credential.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    binding = load_worker_generation(path, CAMERA, now=NOW)
    assert payload["secret"] not in repr(binding)
    for bad in [
        {**payload, "cameraId": str(uuid4())},
        {**payload, "secret": "secret"},
        {**payload, "expiresAt": NOW.isoformat()},
        {**payload, "unknown": True},
    ]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError) as exc:
            load_worker_generation(path, CAMERA, now=NOW)
        assert payload["secret"] not in str(exc.value)
    path.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError):
        load_worker_generation(path, CAMERA, now=NOW)
    path.write_text(json.dumps(payload)[:-1] + ', "cameraId": "' + str(CAMERA) + '"}')
    with pytest.raises(ValueError):
        load_worker_generation(path, CAMERA, now=NOW)


class FakeSource:
    source_id = "synthetic"
    is_connected = False

    def __init__(self):
        self.session = uuid4()
        self.sequence = 0
        self.opens = 0

    async def connect(self):
        self.opens += 1
        self.is_connected = True

    async def close(self):
        self.is_connected = False

    async def read_frame(self):
        frame = FrameEnvelope(
            stream_id="synthetic",
            camera_external_id="CAM",
            session_id=self.session,
            captured_at=NOW,
            sequence_number=self.sequence,
            width=1,
            height=1,
            payload=b"123",
        )
        self.sequence += 1
        return frame


@pytest.mark.anyio
async def test_bound_source_uses_issued_session_once_and_preserves_sequence():
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = FakeSource()
    bound = GenerationBoundSource(source, binding, clock=lambda: NOW)
    await bound.connect()
    for index in range(2):
        frame = await bound.read_frame()
        assert frame.session_id == UUID(binding.stream_session_id)
        assert frame.sequence_number == index
    await bound.close()
    with pytest.raises(RuntimeError):
        await bound.connect()
    assert source.opens == 1


@pytest.mark.anyio
async def test_bound_source_rejects_changed_underlying_session_and_expired_binding():
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = FakeSource()
    current = [NOW]
    bound = GenerationBoundSource(source, binding, clock=lambda: current[0])
    await bound.connect()
    await bound.read_frame()
    source.session = uuid4()
    with pytest.raises(RuntimeError):
        await bound.read_frame()
    current[0] = NOW + timedelta(hours=1)
    with pytest.raises(RuntimeError):
        await bound.read_frame()
    await bound.close()
    assert not source.is_connected


@pytest.mark.anyio
async def test_multi_camera_preflight_consumes_binding_before_model_or_source_open(tmp_path: Path):
    from types import SimpleNamespace

    from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
    from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
    from smartsite_ai.integrations.backend_configuration_client import ConfigurationBundleResult
    from smartsite_ai.runtime.assembly import (
        assemble_sessions,
        close_configuration_clients,
        preflight_manifest,
    )
    from smartsite_ai.runtime.manifest import bind_manifest, parse_manifest

    model = tmp_path / "model.json"
    model.write_text("{}")
    video = tmp_path / "synthetic.mp4"
    video.write_bytes(b"not-video")
    credential = credential_payload()
    credential["expiresAt"] = (
        (datetime.now(UTC) + timedelta(hours=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    file = tmp_path / "binding.json"
    file.write_text(json.dumps(credential))
    file.chmod(0o600)
    manifest = parse_manifest(
        {
            "schemaVersion": "1",
            "modelSpec": str(model),
            "cameras": [
                {
                    "streamId": "synthetic",
                    "cameraId": str(CAMERA),
                    "cameraExternalId": "CAM",
                    "ppeRegionId": str(uuid4()),
                    "source": {"kind": "path", "path": str(video)},
                    "outbox": str(tmp_path / "outbox.sqlite3"),
                    "generationCredentialFile": str(file),
                }
            ],
        }
    )
    bound = bind_manifest(manifest, {})
    content = {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA),
        "cameraExternalId": "CAM",
        "configurationVersion": 1,
        "viewRevision": 1,
        "issuedAt": "2026-10-27T00:00:00.000Z",
        "regions": [],
    }
    bundle = CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode()
    )

    class Client:
        async def fetch_bundle(self, *args, **kwargs):
            return ConfigurationBundleResult(bundle, '"v1"')

        async def aclose(self):
            pass

    failed, plans = await preflight_manifest(bound, configuration_client_factory=Client)
    assert failed is None
    source = FakeSource()
    sessions = assemble_sessions(
        plans,
        SimpleNamespace(),
        backend_client_factory=lambda: SimpleNamespace(),
        source_factory=lambda config: source,
        evidence_max_bytes=1024,
        delivery_interval_seconds=0.25,
    )
    assert source.opens == 0
    stream = sessions[0].worker._stream
    assert stream.config.max_consecutive_failures == 1
    await stream.source.connect()
    assert (await stream.source.read_frame()).session_id == UUID(credential["streamSessionId"])
    await stream.source.close()
    await close_configuration_clients(plans)
    failed, plans = await preflight_manifest(bound, configuration_client_factory=Client)
    assert failed is not None
    assert plans == ()


@pytest.mark.anyio
async def test_single_cli_claim_is_consumed_even_if_model_start_fails(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
    from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
    from smartsite_ai.integrations.backend_configuration_client import ConfigurationBundleResult
    from smartsite_ai.tools import run_camera_worker as cli

    model = tmp_path / "model.json"
    model.write_text("{}")
    payload = credential_payload()
    payload["expiresAt"] = (
        (datetime.now(UTC) + timedelta(hours=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    file = tmp_path / "binding.json"
    file.write_text(json.dumps(payload))
    file.chmod(0o600)
    content = {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA),
        "cameraExternalId": "CAM",
        "configurationVersion": 1,
        "viewRevision": 1,
        "issuedAt": "2026-10-27T00:00:00.000Z",
        "regions": [],
    }
    bundle = CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode()
    )

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def fetch_bundle(self, *args, **kwargs):
            return ConfigurationBundleResult(bundle, '"v1"')

    monkeypatch.setattr(cli.BackendClient, "from_settings", lambda settings: Client())
    monkeypatch.setattr(cli.BackendConfigurationClient, "from_settings", lambda settings: Client())
    calls = []

    async def refuse_model(*args, **kwargs):
        calls.append("model-boundary-only")
        raise RuntimeError("Synthetic pre-model stop")

    monkeypatch.setattr(cli, "load_warmed_yolo11_detector", refuse_model)
    args = cli.build_parser().parse_args(
        [
            "--source",
            "0",
            "--stream-id",
            "synthetic",
            "--camera-id",
            str(CAMERA),
            "--camera-external-id",
            "CAM",
            "--ppe-region-id",
            str(uuid4()),
            "--model-spec",
            str(model),
            "--outbox",
            str(tmp_path / "outbox.sqlite3"),
            "--generation-credential-file",
            str(file),
        ]
    )
    with pytest.raises(RuntimeError, match="Synthetic pre-model stop"):
        await cli.run_worker(args, settings=SimpleNamespace())
    with pytest.raises(RuntimeError, match="already consumed"):
        await cli.run_worker(args, settings=SimpleNamespace())
    assert calls == ["model-boundary-only"]


@pytest.mark.anyio
async def test_actual_stream_worker_stops_bound_generation_on_connection_loss():
    from smartsite_ai.ingestion.config import StreamConfig
    from smartsite_ai.ingestion.source import SourceReadError
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.ingestion.worker import StreamWorker
    from smartsite_ai.integrations.worker_generation import (
        GenerationBoundSource,
        WorkerGenerationCredential,
    )

    class LostSource(FakeSource):
        async def read_frame(self):
            await asyncio.sleep(0)
            if self.sequence:
                raise SourceReadError("Synthetic connection loss")
            return await super().read_frame()

    binding = WorkerGenerationCredential.model_validate(credential_payload())
    source = LostSource()
    stream = StreamWorker(
        StreamConfig(
            stream_id="synthetic",
            camera_external_id="CAM",
            source_url="fake://synthetic",
            max_consecutive_failures=1,
        ),
        source=GenerationBoundSource(source, binding, clock=lambda: NOW),
    )
    await stream.start()
    await asyncio.wait_for(stream._loop_task, 2)
    assert stream.state is StreamState.ERROR
    assert source.opens == 1
    assert not source.is_connected
    await stream.stop()
