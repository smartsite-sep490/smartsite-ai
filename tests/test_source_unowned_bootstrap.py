from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from smartsite_ai.source_activation import (
    Scope,
    SourceActivationError,
    SourceActivator,
    SourceRegistry,
)
from smartsite_ai.source_control_api import source_control_router
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_process_driver import SourceProcessDriver
from smartsite_ai.tools import run_source_control as control

SCOPE: Scope = ("11111111-1111-4111-8111-111111111111", "camera-a")
CMD1 = "22222222-2222-4222-8222-222222222222"
CMD2 = "33333333-3333-4333-8333-333333333333"
RECEIPT1 = "44444444-4444-4444-8444-444444444444"
RECEIPT2 = "55555555-5555-4555-8555-555555555555"
GEN1 = "66666666-6666-4666-8666-666666666666"
GEN2 = "77777777-7777-4777-8777-777777777777"
SESS1 = "88888888-8888-4888-8888-888888888888"
SESS2 = "99999999-9999-4999-8999-999999999999"


def _generation(
    worker_generation_id: str = GEN1,
    stream_session_id: str = SESS1,
    *,
    secret: str | None = None,
) -> dict[str, str]:
    return {
        "workerGenerationId": worker_generation_id,
        "streamSessionId": stream_session_id,
        "secret": secret or ("s" * 48),
        "expiresAt": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    }


def _write_stop_aware_child(script: Path, marker: Path, stop_file: Path) -> None:
    script.write_text(
        "import json, os, pathlib, time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        f"stop_file = pathlib.Path({str(stop_file)!r})\n"
        "marker.write_text(os.environ['SMARTSITE_AI_SOURCE_RUNTIME_GENERATION'])\n"
        "while not stop_file.exists():\n"
        "    time.sleep(0.02)\n",
        encoding="utf-8",
    )


class _UnownedFakeDriver:
    def __init__(
        self,
        *,
        initial_owner: str | None = None,
        stop_error: Exception | None = None,
        start_error: Exception | None = None,
        stays_running_on_start: bool = True,
    ) -> None:
        self.owner = initial_owner
        self.stop_error = stop_error
        self.start_error = start_error
        self.stays_running_on_start = stays_running_on_start
        self.calls: list[object] = []
        self.running = initial_owner is not None

    def supports(self, scope: Scope) -> bool:
        return scope == SCOPE

    async def generation_id(self, scope: Scope) -> str | None:
        if not self.supports(scope):
            raise SourceActivationError()
        self.calls.append(("generation_id", scope))
        return self.owner

    async def stop(self, scope: Scope) -> None:
        self.calls.append(("stop", scope))
        if self.stop_error is not None:
            raise self.stop_error
        self.running = False

    async def start(self, scope: Scope, source: str, generation: Mapping[str, str]) -> None:
        assert not self.running
        self.calls.append(("start", source, generation["workerGenerationId"]))
        if self.start_error is not None:
            raise self.start_error
        self.owner = generation["workerGenerationId"]
        self.running = self.stays_running_on_start

    async def is_running(self, scope: Scope, generation_id: str) -> bool:
        return self.running and self.owner == generation_id


def test_source_process_driver_seed_requirement_and_allow_unowned_start_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_seed = tmp_path / "missing-generation.json"
    stop_file = tmp_path / "STOP"
    marker = tmp_path / "ready.json"
    child = tmp_path / "child.py"
    _write_stop_aware_child(child, marker, stop_file)

    async def ready(identifier: str) -> bool:
        return (
            marker.exists()
            and json.loads(marker.read_text(encoding="utf-8"))["workerGenerationId"] == identifier
        )

    async def run() -> None:
        default_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=missing_seed,
            environment={},
            ready=ready,
        )
        with pytest.raises((FileNotFoundError, SourceActivationError)):
            await default_driver.generation_id(SCOPE)

        explicit_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=missing_seed,
            environment={},
            ready=ready,
            allow_unowned_start=True,
        )
        assert await explicit_driver.generation_id(SCOPE) is None

        for foreign in [("foreign-site", SCOPE[1]), (SCOPE[0], "foreign-camera")]:
            with pytest.raises(SourceActivationError):
                await explicit_driver.generation_id(foreign)

        # In-memory owner wins over absent seed, and clean stop returns None.
        await explicit_driver.stop(SCOPE)
        gen = _generation(GEN1, SESS1)
        await explicit_driver.start(SCOPE, "0", gen)
        try:
            assert await explicit_driver.is_running(SCOPE, GEN1)
            assert await explicit_driver.generation_id(SCOPE) == GEN1
        finally:
            await explicit_driver.stop(SCOPE)
        assert await explicit_driver.generation_id(SCOPE) is None

        # Present corrupt seed file must still reject even with allow_unowned_start=True.
        corrupt_seed = tmp_path / "corrupt-generation.json"
        corrupt_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=corrupt_seed,
            environment={},
            ready=ready,
            allow_unowned_start=True,
        )
        for bad_bytes in (b"{not-json", b"{}"):
            corrupt_seed.write_bytes(bad_bytes)
            with pytest.raises((ValueError, KeyError, SourceActivationError)):
                await corrupt_driver.generation_id(SCOPE)

        # In-memory owner still wins even when seed file holds a different ID.
        corrupt_seed.write_text(json.dumps({"workerGenerationId": str(uuid4())}), encoding="utf-8")
        await corrupt_driver.start(SCOPE, "0", gen)
        try:
            assert await corrupt_driver.generation_id(SCOPE) == GEN1
        finally:
            await corrupt_driver.stop(SCOPE)

        # Existing but unreadable seed file must not fall back to unowned None.
        unreadable_seed = tmp_path / "unreadable-generation.json"
        unreadable_seed.write_text(json.dumps({"workerGenerationId": GEN2}), encoding="utf-8")
        real_read_bytes = Path.read_bytes
        real_open = Path.open

        def guarded_read_bytes(self: Path) -> bytes:
            if self == unreadable_seed:
                raise PermissionError("unreadable seed")
            return real_read_bytes(self)

        def guarded_open(self: Path, *args: object, **kwargs: object) -> object:
            if self == unreadable_seed:
                raise PermissionError("unreadable seed")
            return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
        monkeypatch.setattr(Path, "open", guarded_open)
        unreadable_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=stop_file,
            generation_file=unreadable_seed,
            environment={},
            ready=ready,
            allow_unowned_start=True,
        )
        with pytest.raises((PermissionError, OSError, SourceActivationError)):
            await unreadable_driver.generation_id(SCOPE)

    asyncio.run(run())


def test_source_process_driver_unowned_blocks_reconciliation_latch_and_present_unowned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_seed = tmp_path / "missing-generation.json"

    class _FakeHungProcess:
        def __init__(self) -> None:
            self.pid = 7272
            self.returncode = None
            self._terminated = asyncio.Event()

        def kill(self) -> None:
            self.returncode = -9
            self._terminated.set()

        async def wait(self) -> int | None:
            await self._terminated.wait()
            return self.returncode

    async def fake_exec(*_cmd: object, **_kwargs: object) -> _FakeHungProcess:
        return _FakeHungProcess()

    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        asyncio, "timeout", lambda d: real_timeout(0.02 if d is not None and d >= 1 else d)
    )

    async def ready(_identifier: str) -> bool:
        return True

    async def run() -> None:
        latched_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(tmp_path / "capture-cli"),),
            stop_file=tmp_path / "STOP-latched",
            generation_file=missing_seed,
            environment={},
            ready=ready,
            allow_unowned_start=True,
        )
        await latched_driver.start(SCOPE, "0", _generation(GEN1, SESS1))
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await latched_driver.stop(SCOPE)
        with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
            await latched_driver.generation_id(SCOPE)

        # Present child without an owned generation must never degrade to unowned None.
        orphan_driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(tmp_path / "capture-cli"),),
            stop_file=tmp_path / "STOP-orphan",
            generation_file=missing_seed,
            environment={},
            ready=ready,
            allow_unowned_start=True,
        )
        orphan_driver.process = _FakeHungProcess()  # type: ignore[assignment]
        with pytest.raises((FileNotFoundError, SourceActivationError)):
            await orphan_driver.generation_id(SCOPE)

    asyncio.run(run())


def test_source_process_driver_dangling_symlink_seed_fails_closed(tmp_path: Path) -> None:
    async def ready(_identifier: str) -> bool:
        return True

    symlink_seed = tmp_path / "dangling-generation.json"
    missing_target = tmp_path / "nonexistent-generation.json"
    try:
        symlink_seed.symlink_to(missing_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"OS symlink creation unavailable in this environment: {exc}")

    driver = SourceProcessDriver(
        scope=SCOPE,
        command=(str(tmp_path / "capture-cli"),),
        stop_file=tmp_path / "STOP-symlink",
        generation_file=symlink_seed,
        environment={},
        ready=ready,
        allow_unowned_start=True,
    )

    async def run() -> None:
        with pytest.raises((FileNotFoundError, OSError, SourceActivationError)):
            await driver.generation_id(SCOPE)

    asyncio.run(run())


def test_source_activator_unowned_bootstrap_persists_uncertainty_and_is_idempotent(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        key = SecretStr(Fernet.generate_key().decode())
        failing_path = tmp_path / "failing.enc"
        failing_store = SourceRegistry(failing_path, key)
        failing_activator = SourceActivator(
            failing_store,
            _UnownedFakeDriver(initial_owner=None, stop_error=RuntimeError("stop-interrupted")),
        )
        failing_activator.remember(
            SCOPE, RECEIPT1, "rtsp://op:super-secret@192.168.1.50/live", "LIVE"
        )
        with pytest.raises(RuntimeError, match="stop-interrupted"):
            await failing_activator.prepare(SCOPE, CMD1, RECEIPT1, 0)
        persisted_uncertain = failing_store.read(SCOPE)
        assert persisted_uncertain is not None
        assert persisted_uncertain["state"] == "STOPPING"
        assert persisted_uncertain["currentGenerationId"] is None
        assert b"super-secret" not in failing_path.read_bytes()

        # Nonzero expected revision with absent registry rejects before driver lookup or stop.
        nonzero_driver = _UnownedFakeDriver(initial_owner=None)
        nonzero_activator = SourceActivator(
            SourceRegistry(tmp_path / "rev.enc", key), nonzero_driver
        )
        nonzero_activator.remember(SCOPE, RECEIPT1, "0", "LIVE")
        with pytest.raises(SourceActivationError):
            await nonzero_activator.prepare(SCOPE, CMD1, RECEIPT1, 1)
        assert nonzero_driver.calls == []

        # Absent registry + expected_revision=0 accepts None -> PREPARED -> RUNNING.
        ok_path = tmp_path / "ok.enc"
        ok_driver = _UnownedFakeDriver(initial_owner=None)
        ok_activator = SourceActivator(SourceRegistry(ok_path, key), ok_driver)
        ok_activator.remember(SCOPE, RECEIPT1, "rtsp://op:child-secret@192.168.1.50/live", "LIVE")

        prepared = await ok_activator.prepare(SCOPE, CMD1, RECEIPT1, 0)
        assert prepared == {
            "state": "PREPARED",
            "commandId": CMD1,
            "sourceRevision": 0,
            "sourceKind": "LIVE",
            "currentGenerationId": None,
        }
        assert ok_driver.calls == [("generation_id", SCOPE), ("stop", SCOPE)]

        restarted_activator = SourceActivator(SourceRegistry(ok_path, key), ok_driver)
        assert await restarted_activator.prepare(SCOPE, CMD1, RECEIPT1, 0) == prepared
        assert len(ok_driver.calls) == 2

        with pytest.raises(SourceActivationError):
            await restarted_activator.prepare(SCOPE, CMD1, RECEIPT2, 0)

        gen = _generation(GEN1, SESS1, secret="backend-secret-" + "x" * 32)
        running = await restarted_activator.commit(SCOPE, CMD1, gen)
        assert running == {
            "state": "RUNNING",
            "commandId": CMD1,
            "sourceRevision": 1,
            "sourceKind": "LIVE",
            "currentGenerationId": GEN1,
        }
        assert "backend-secret-" not in repr(running)
        assert "child-secret" not in repr(running)
        raw_ciphertext = ok_path.read_bytes()
        assert b"backend-secret-" not in raw_ciphertext and b"child-secret" not in raw_ciphertext

        assert await restarted_activator.prepare(SCOPE, CMD1, RECEIPT1, 0) == running
        assert await restarted_activator.commit(SCOPE, CMD1, gen) == running

    asyncio.run(run())


def test_source_activator_rejects_none_generation_when_prior_or_recovery_row_exists(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        key = SecretStr(Fernet.generate_key().decode())
        for label, prior_row, expected_rev in [
            (
                "running-rev1",
                {
                    "state": "RUNNING",
                    "commandId": CMD1,
                    "sourceRevision": 1,
                    "sourceKind": "LIVE",
                    "source": "0",
                    "currentGenerationId": GEN1,
                },
                1,
            ),
            (
                "stopped-rev0",
                {
                    "state": "STOPPED",
                    "commandId": CMD1,
                    "sourceRevision": 0,
                    "sourceKind": "LIVE",
                    "source": "0",
                    "currentGenerationId": GEN1,
                },
                0,
            ),
            (
                "prepared-other-cmd",
                {
                    "state": "PREPARED",
                    "commandId": CMD1,
                    "sourceRevision": 0,
                    "sourceKind": "LIVE",
                    "source": "0",
                    "currentGenerationId": None,
                },
                0,
            ),
            (
                "stopping-recovery",
                {
                    "state": "STOPPING",
                    "commandId": CMD1,
                    "sourceRevision": 0,
                    "sourceKind": "LIVE",
                    "source": "0",
                    "currentGenerationId": None,
                },
                0,
            ),
        ]:
            store = SourceRegistry(tmp_path / f"{label}.enc", key)
            store.write(SCOPE, dict(prior_row))
            driver = _UnownedFakeDriver(initial_owner=None)
            activator = SourceActivator(store, driver)
            activator.remember(SCOPE, RECEIPT2, "0", "LIVE")
            with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
                await activator.prepare(SCOPE, CMD2, RECEIPT2, expected_rev)
            assert all(call[0] != "stop" for call in driver.calls), label
            assert store.read(SCOPE) == prior_row, label

    asyncio.run(run())


def test_source_control_api_unowned_prepare_v11_and_running_v10_contract(
    tmp_path: Path,
) -> None:
    key = SecretStr(Fernet.generate_key().decode())
    store = SourceRegistry(tmp_path / "api.enc", key)
    driver = _UnownedFakeDriver(initial_owner=None)
    activator = SourceActivator(store, driver)

    async def probe(source: str, live: bool, scope: Scope) -> dict[str, object]:
        return {
            "width": 640,
            "height": 480,
            "capturedAt": datetime.now(UTC).isoformat(),
            "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
        }

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
            activator=activator,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + "a" * 48}
    base = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"

    receipt1 = client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    cmd1_body = {"commandId": CMD1, "receiptId": receipt1, "expectedRevision": 0}

    assert client.post(f"{base}/prepare", json=cmd1_body).status_code == 401
    assert (
        client.post(
            f"/v1/source-control/{uuid4()}/{SCOPE[1]}/prepare", headers=headers, json=cmd1_body
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"{base}/prepare", headers=headers, json={**cmd1_body, "source": "arbitrary"}
        ).status_code
        == 400
    )

    prep1 = client.post(f"{base}/prepare", headers=headers, json=cmd1_body)
    assert prep1.status_code == 200
    assert prep1.json() == {
        "schemaVersion": "1.1.0",
        "state": "PREPARED",
        "commandId": CMD1,
        "sourceRevision": 0,
        "sourceKind": "LIVE",
        "currentGenerationId": None,
    }
    assert client.post(f"{base}/prepare", headers=headers, json=cmd1_body).json() == prep1.json()

    resume_on_unowned = client.post(
        f"{base}/prepare-resume",
        headers=headers,
        json={
            "commandId": CMD2,
            "expectedRevision": 0,
            "generationBinding": {
                "cameraId": str(uuid4()),
                "workerGenerationId": GEN1,
                "streamSessionId": SESS1,
            },
            "configuration": {
                "configurationVersion": 1,
                "viewRevision": 1,
                "bundleSha256": "a" * 64,
            },
        },
    )
    assert resume_on_unowned.status_code == 409

    gen1 = _generation(GEN1, SESS1, secret="first-secret-" + "b" * 35)
    commit1 = client.post(
        f"{base}/commit", headers=headers, json={"commandId": CMD1, "generation": gen1}
    )
    assert commit1.status_code == 200
    assert commit1.json() == {
        "schemaVersion": "1.0.0",
        "state": "RUNNING",
        "commandId": CMD1,
        "sourceRevision": 1,
        "sourceKind": "LIVE",
        "currentGenerationId": GEN1,
    }
    assert "first-secret-" not in commit1.text
    assert client.post(f"{base}/prepare", headers=headers, json=cmd1_body).json() == commit1.json()

    receipt2 = client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    prep2 = client.post(
        f"{base}/prepare",
        headers=headers,
        json={"commandId": CMD2, "receiptId": receipt2, "expectedRevision": 1},
    )
    assert prep2.status_code == 200
    assert prep2.json() == {
        "schemaVersion": "1.0.0",
        "state": "PREPARED",
        "commandId": CMD2,
        "sourceRevision": 1,
        "sourceKind": "LIVE",
        "currentGenerationId": GEN1,
    }
    commit2 = client.post(
        f"{base}/commit",
        headers=headers,
        json={"commandId": CMD2, "generation": _generation(GEN2, SESS2)},
    )
    assert commit2.status_code == 200
    assert commit2.json()["schemaVersion"] == "1.0.0"
    assert commit2.json()["state"] == "RUNNING"
    assert commit2.json()["sourceRevision"] == 2
    assert commit2.json()["currentGenerationId"] == GEN2


def test_source_control_api_partial_commit_stays_reconciliation_required(
    tmp_path: Path,
) -> None:
    key = SecretStr(Fernet.generate_key().decode())
    store = SourceRegistry(tmp_path / "partial.enc", key)
    driver = _UnownedFakeDriver(initial_owner=None, stays_running_on_start=False)
    activator = SourceActivator(store, driver)

    async def probe(source: str, live: bool, scope: Scope) -> dict[str, object]:
        return {
            "width": 640,
            "height": 480,
            "capturedAt": datetime.now(UTC).isoformat(),
            "imageDataUrl": "data:image/jpeg;base64,/9j/2Q==",
        }

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
            activator=activator,
        )
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + "a" * 48}
    base = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"

    receipt1 = client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    prep = client.post(
        f"{base}/prepare",
        headers=headers,
        json={"commandId": CMD1, "receiptId": receipt1, "expectedRevision": 0},
    )
    assert prep.status_code == 200 and prep.json()["schemaVersion"] == "1.1.0"

    failed_commit = client.post(
        f"{base}/commit",
        headers=headers,
        json={"commandId": CMD1, "generation": _generation(GEN1, SESS1)},
    )
    assert failed_commit.status_code in {409, 503}
    assert store.read(SCOPE)["state"] == "STARTING"  # type: ignore[index]

    # Even if driver later reports None after failed commit, follow-up fails closed.
    driver.owner = None
    driver.running = False
    receipt2 = client.post(
        f"{base}/probe", headers=headers, json={"kind": "WEBCAM", "deviceId": "usb0"}
    ).json()["receiptId"]
    blocked_followup = client.post(
        f"{base}/prepare",
        headers=headers,
        json={"commandId": CMD2, "receiptId": receipt2, "expectedRevision": 0},
    )
    assert blocked_followup.status_code == 409


def test_default_source_control_cli_retains_apply_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "host-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "host_id": "edge",
                "label": "Host",
                "scopes": [[SCOPE[0], SCOPE[1]]],
                "webcams": {"usb0": 0},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_uvicorn_run(app: FastAPI, *, host: str, port: int, **_kwargs: object) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(control.uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setenv("SMARTSITE_AI_SOURCE_CONTROL_TOKEN", "a" * 48)
    monkeypatch.setattr(sys, "argv", ["run_source_control", "--manifest", str(manifest_path)])
    control.main()

    app = captured["app"]
    assert isinstance(app, FastAPI)
    assert captured["host"] == "127.0.0.1"
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + "a" * 48}
    base = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"
    caps = client.get(f"{base}/capabilities", headers=headers)
    assert caps.status_code == 200
    assert caps.json()["applySupported"] is False
    assert (
        client.post(
            f"{base}/prepare",
            headers=headers,
            json={"commandId": CMD1, "receiptId": RECEIPT1, "expectedRevision": 0},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"{base}/commit",
            headers=headers,
            json={"commandId": CMD1, "generation": _generation(GEN1, SESS1)},
        ).status_code
        == 404
    )
