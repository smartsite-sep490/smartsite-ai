import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.integrations.backend_configuration_client import ConfigurationSnapshotResult
from smartsite_ai.integrations.outbox import SqliteEventOutbox

CAMERA = UUID("11111111-1111-4111-8111-111111111111")
REGION = UUID("22222222-2222-4222-8222-222222222222")


def credential_file(tmp_path: Path):
    document = {
        "schemaVersion": "1.0.0",
        "cameraId": str(CAMERA),
        "workerGenerationId": str(uuid4()),
        "streamSessionId": str(uuid4()),
        "secret": "x" * 43,
        "expiresAt": (datetime.now(UTC) + timedelta(hours=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
    file = tmp_path / "credential.json"
    file.write_text(json.dumps(document))
    file.chmod(0o600)
    return file, document


class ConfigurationClient:
    closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def aclose(self):
        self.closed = True

    async def fetch(self, *args, **kwargs):
        snapshot = CameraRegionConfiguration.from_wire_bytes(
            json.dumps(
                {
                    "schemaVersion": "1.0.0",
                    "configurationVersion": 1,
                    "cameraExternalId": "CAM",
                    "regions": [
                        {
                            "regionId": str(REGION),
                            "geometryVersion": 1,
                            "coordinateSpace": "NORMALIZED_0_1",
                            "polygon": {"coordinates": [[0, 0], [1, 0], [0, 1]]},
                        }
                    ],
                }
            ).encode()
        )
        return ConfigurationSnapshotResult(snapshot, '"v1"')


def test_durable_claim_is_unique_across_instances_and_restart(tmp_path: Path):
    async def scenario():
        path = tmp_path / "outbox.sqlite3"
        generation, session = uuid4(), uuid4()
        results = await asyncio.gather(
            *(
                SqliteEventOutbox(path).claim_worker_session(CAMERA, generation, session)
                for _ in range(10)
            ),
            return_exceptions=True,
        )
        assert sum(result is None for result in results) == 1
        assert all(result is None or isinstance(result, RuntimeError) for result in results), [
            (type(result).__name__, str(result)) for result in results if result is not None
        ]
        restarted = SqliteEventOutbox(path)
        for gen, sess in [(generation, uuid4()), (uuid4(), session)]:
            with pytest.raises(RuntimeError, match="already consumed"):
                await restarted.claim_worker_session(CAMERA, gen, sess)
        await restarted.claim_worker_session(CAMERA, uuid4(), uuid4())

    asyncio.run(scenario())


def test_initialization_retries_only_bounded_sqlite_contention(tmp_path: Path, monkeypatch):
    outbox = SqliteEventOutbox(tmp_path / "outbox.sqlite3")
    original = outbox._connect
    calls = []

    def connect():
        calls.append(None)
        if len(calls) == 1:
            error = sqlite3.OperationalError("synthetic contention")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return original()

    monkeypatch.setattr(outbox, "_connect", connect)
    asyncio.run(outbox.initialize())
    assert len(calls) == 2

    for code, expected in [(sqlite3.SQLITE_LOCKED, 4), (sqlite3.SQLITE_ERROR, 1)]:
        failed = SqliteEventOutbox(tmp_path / f"failed-{code}.sqlite3")
        attempts = []

        def fail(attempts=attempts, code=code):
            attempts.append(None)
            error = sqlite3.OperationalError("synthetic failure")
            error.sqlite_errorcode = code
            raise error

        monkeypatch.setattr(failed, "_connect", fail)
        with pytest.raises(sqlite3.OperationalError):
            asyncio.run(failed.initialize())
        assert len(attempts) == expected


def test_separate_processes_cannot_consume_one_generation_twice(tmp_path: Path):
    async def scenario():
        path = tmp_path / "process-outbox.sqlite3"
        generation, session = uuid4(), uuid4()
        program = """
import asyncio, sys
from pathlib import Path
from uuid import UUID
from smartsite_ai.integrations.outbox import SqliteEventOutbox
async def main():
    try:
        outbox = SqliteEventOutbox(Path(sys.argv[1]))
        await outbox.claim_worker_session(*(UUID(v) for v in sys.argv[2:]))
    except RuntimeError:
        return 2
    return 0
raise SystemExit(asyncio.run(main()))
"""
        processes = await asyncio.gather(
            *(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    program,
                    str(path),
                    str(CAMERA),
                    str(generation),
                    str(session),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                for _ in range(4)
            )
        )
        outputs = await asyncio.gather(*(process.communicate() for process in processes))
        assert all(out == b"" and err == b"" for out, err in outputs)
        assert sorted(process.returncode for process in processes) == [0, 2, 2, 2]
        with sqlite3.connect(path) as connection:
            assert (
                connection.execute("SELECT count(*) FROM worker_session_claim").fetchone()[0] == 1
            )
            assert [
                row[1] for row in connection.execute("PRAGMA table_info(worker_session_claim)")
            ] == ["generation_id", "session_id", "camera_id", "claimed_at"]

    asyncio.run(scenario())


def test_multi_preflight_claims_before_source_and_duplicate_preflight_fails(tmp_path: Path):
    async def scenario():
        from smartsite_ai.runtime.assembly import (
            assemble_sessions,
            close_configuration_clients,
            preflight_manifest,
        )
        from smartsite_ai.runtime.manifest import bind_manifest, parse_manifest

        file, document = credential_file(tmp_path)
        model = tmp_path / "model.json"
        model.write_text("{}")
        video = tmp_path / "synthetic.mp4"
        video.write_bytes(b"not-video")
        manifest = parse_manifest(
            {
                "schemaVersion": "1",
                "modelSpec": str(model),
                "cameras": [
                    {
                        "streamId": "synthetic",
                        "cameraId": str(CAMERA),
                        "cameraExternalId": "CAM",
                        "ppeRegionId": str(REGION),
                        "source": {"kind": "path", "path": str(video)},
                        "live": True,
                        "outbox": str(tmp_path / "outbox.sqlite3"),
                        "generationCredentialFile": str(file),
                    }
                ],
            }
        )
        bound = bind_manifest(manifest, {})
        clients = []

        def client():
            result = ConfigurationClient()
            clients.append(result)
            return result

        failed, plans = await preflight_manifest(bound, configuration_client_factory=client)
        assert failed is None
        source = SimpleNamespace(source_id="synthetic", is_connected=False)
        sessions = assemble_sessions(
            plans,
            SimpleNamespace(),
            backend_client_factory=lambda: SimpleNamespace(),
            source_factory=lambda config: source,
            evidence_max_bytes=1024,
            delivery_interval_seconds=0.25,
        )
        stream = sessions[0].worker._stream
        assert stream.source.binding.stream_session_id == document["streamSessionId"]
        assert stream.config.max_consecutive_failures == 1
        assert not source.is_connected
        await close_configuration_clients(plans)
        failed, plans = await preflight_manifest(bound, configuration_client_factory=client)
        assert failed is not None
        assert plans == ()
        assert all(c.closed for c in clients)

    asyncio.run(scenario())


def test_single_cli_consumes_claim_before_mocked_model_failure(tmp_path: Path, monkeypatch):
    async def scenario():
        from smartsite_ai.tools import run_camera_worker as cli

        file, _ = credential_file(tmp_path)
        model = tmp_path / "model.json"
        model.write_text("{}")
        monkeypatch.setattr(
            cli.BackendClient, "from_settings", lambda settings: ConfigurationClient()
        )
        monkeypatch.setattr(
            cli.BackendConfigurationClient, "from_settings", lambda settings: ConfigurationClient()
        )
        calls = []

        def fail_model(*args, **kwargs):
            calls.append("model-boundary")
            raise RuntimeError("Synthetic model boundary stop")

        monkeypatch.setattr(cli, "load_yolo11_detector", fail_model)
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
                str(REGION),
                "--model-spec",
                str(model),
                "--outbox",
                str(tmp_path / "outbox.sqlite3"),
                "--generation-credential-file",
                str(file),
            ]
        )
        with pytest.raises(RuntimeError, match="Synthetic model boundary stop"):
            await cli.run_worker(args, settings=SimpleNamespace())
        with pytest.raises(RuntimeError, match="already consumed"):
            await cli.run_worker(args, settings=SimpleNamespace())
        assert calls == ["model-boundary"]

    asyncio.run(scenario())
