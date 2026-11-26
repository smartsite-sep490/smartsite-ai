"""Bounded TDD tests for Backend-commanded stopped-owner resume recovery."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from pydantic import SecretStr
from test_source_operator import CONTROL_TOKEN, _operator_bundle, _valid_image_payload
from test_source_resume import OLD as LEGACY_OLD
from test_source_resume import SCOPE as LEGACY_SCOPE
from test_source_resume import Driver as LegacyResumeDriver
from test_source_resume import request as legacy_request
from test_tracked_policy_pipeline import policy_bundle

from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.source_activation import (
    Scope,
    SourceActivationError,
    SourceActivator,
    SourceRegistry,
    StoppedOwnerResumeDriver,
)
from smartsite_ai.source_control_api import source_control_router
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_process_driver import SourceProcessDriver
from smartsite_ai.source_resume_models import (
    ResumeBinding,
    ResumeConfiguration,
    ResumeManifest,
    ResumePrepare,
)
from smartsite_ai.tools import source_operator as op

SCOPE: Scope = ("11111111-1111-4111-8111-111111111111", "camera-a")
CAMERA_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
RESOURCE_ID = UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")
GEN1 = "66666666-6666-4666-8666-666666666666"
GEN2 = "77777777-7777-4777-8777-777777777777"
GEN3 = "88888888-8888-4888-8888-888888888888"
SESS1 = "12121212-1212-4121-8121-121212121212"
SESS2 = "23232323-2323-4232-8232-232323232323"
SESS3 = "34343434-3434-4343-8343-343434343434"
CMD_OLD = "45454545-4545-4454-8454-454545454545"
CMD_NEW = "56565656-5656-4565-8565-565656565656"
CMD_3 = "67676767-6767-4676-8676-676767676767"
CONFIG = ResumeConfiguration(configurationVersion=12, viewRevision=7, bundleSha256="a" * 64)


def _generation(
    worker_generation_id: str,
    stream_session_id: str,
    *,
    expired: bool = False,
    secret: str | None = None,
) -> dict[str, str]:
    delta = timedelta(hours=-2) if expired else timedelta(hours=12)
    return {
        "workerGenerationId": worker_generation_id,
        "streamSessionId": stream_session_id,
        "secret": secret or ("s" * 43),
        "expiresAt": (datetime.now(UTC) + delta).isoformat(),
    }


def _write_stop_aware_child(script: Path, marker: Path, stop_file: Path) -> None:
    script.write_text(
        "import json, os, pathlib, time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        f"stop_file = pathlib.Path({str(stop_file)!r})\n"
        "payload = {\n"
        "    'source': os.environ.get('SMARTSITE_AI_SOURCE_RUNTIME_SOURCE'),\n"
        "    'generation': json.loads(os.environ['SMARTSITE_AI_SOURCE_RUNTIME_GENERATION']),\n"
        "    'expected': os.environ.get('SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION'),\n"
        "}\n"
        "marker.write_text(json.dumps(payload), encoding='utf-8')\n"
        "while not stop_file.exists():\n"
        "    time.sleep(0.02)\n",
        encoding="utf-8",
    )


def _driver(
    *,
    child_script: Path,
    marker: Path,
    lock_dir: Path,
    generation_file: Path,
    stop_file: Path,
) -> SourceProcessDriver:
    async def ready(generation_id: str) -> bool:
        if not marker.exists():
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload["generation"]["workerGenerationId"] == generation_id

    async def load(
        _manifest: ResumeManifest, _generation: Mapping[str, str]
    ) -> ResumeConfiguration:
        return CONFIG

    return SourceProcessDriver(
        scope=SCOPE,
        command=(str(Path(sys.executable)), "-I", str(child_script)),
        resume_command=(str(Path(sys.executable)), "-I", str(child_script)),
        stop_file=stop_file,
        generation_file=generation_file,
        environment={},
        ready=ready,
        allow_unowned_start=True,
        resume_camera_id=CAMERA_ID,
        resume_manifest_supported=True,
        resume_configuration_loader=load,
        capture_lease_directory=lock_dir,
        capture_lease_resource_id=RESOURCE_ID,
    )


def _make_async_api(activator: SourceActivator) -> tuple[httpx.AsyncClient, dict[str, str], str]:
    async def probe(*_args: object) -> object:
        raise AssertionError("Resume must never probe a source")

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({}, {}, ()),
            service_token=SecretStr(CONTROL_TOKEN),
            probe=probe,  # type: ignore[arg-type]
            activator=activator,
        )
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
    return (
        client,
        {"Authorization": f"Bearer {CONTROL_TOKEN}"},
        f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}",
    )


def _prepare_request(
    *,
    command_id: str = CMD_NEW,
    worker_generation_id: str = GEN1,
    stream_session_id: str = SESS1,
    expected_revision: int = 1,
    camera_id: str = CAMERA_ID,
) -> ResumePrepare:
    return ResumePrepare(
        commandId=command_id,
        expectedRevision=expected_revision,
        generationBinding=ResumeBinding(
            cameraId=camera_id,
            workerGenerationId=worker_generation_id,
            streamSessionId=stream_session_id,
        ),
        configuration=CONFIG,
    )


def _seed_persisted_running(
    registry: SourceRegistry,
    generation_file: Path,
    *,
    state: str = "RUNNING",
    source_kind: str = "LIVE",
    source_revision: int = 1,
    generation: Mapping[str, str] | None = None,
    with_resume_manifest: bool = True,
    command_id: str = CMD_OLD,
    include_gen_file: bool = True,
) -> None:
    gen = dict(generation or _generation(GEN1, SESS1, expired=True))
    row: dict[str, object] = {
        "state": state,
        "commandId": command_id,
        "requestHash": "f" * 64,
        "sourceRevision": source_revision,
        "sourceKind": source_kind,
        "source": "0",
        "currentGenerationId": GEN1,
        "generation": gen,
    }
    if with_resume_manifest:
        prior_req = _prepare_request(
            command_id=command_id,
            worker_generation_id=GEN1,
            stream_session_id=SESS1,
            expected_revision=source_revision,
        )
        manifest = ResumeManifest(
            generationBinding=ResumeBinding(
                cameraId=CAMERA_ID,
                workerGenerationId=GEN1,
                streamSessionId=SESS1,
            ),
            configuration=CONFIG,
        )
        row.update(
            operation="RESUME",
            resumeRequest=prior_req.model_dump(),
            resumeHistory={command_id: prior_req.model_dump()},
            resumeManifest=manifest.model_dump(),
        )
    registry.write(SCOPE, row)
    if include_gen_file:
        generation_file.write_text(json.dumps({"workerGenerationId": GEN1}), encoding="utf-8")


@pytest.mark.anyio
async def test_source_process_driver_validates_capture_lease_pair_and_writes_metadata_only(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    legacy_child = tmp_path / "legacy-child.py"
    legacy_marker = tmp_path / "legacy-marker.json"
    legacy_stop = tmp_path / "LEGACY-STOP"
    _write_stop_aware_child(legacy_child, legacy_marker, legacy_stop)

    async def load(
        _manifest: ResumeManifest, _generation: Mapping[str, str]
    ) -> ResumeConfiguration:
        return CONFIG

    # Partial or invalid capture lease configuration must fail closed cleanly.
    invalid_pairs: list[tuple[Path | None, UUID | None]] = [
        (lock_dir, None),
        (None, RESOURCE_ID),
        (Path("relative/locks"), RESOURCE_ID),
        (Path("\\\\server\\share\\locks"), RESOURCE_ID),
        (lock_dir, "not-a-uuid"),  # type: ignore[list-item]
    ]
    for bad_dir, bad_id in invalid_pairs:
        with pytest.raises(ValueError):
            SourceProcessDriver(
                scope=SCOPE,
                command=(str(Path(sys.executable)), "-I", str(legacy_child)),
                stop_file=legacy_stop,
                generation_file=tmp_path / "bad-gen.json",
                environment={},
                ready=lambda _gid: _always_false(),
                capture_lease_directory=bad_dir,
                capture_lease_resource_id=bad_id,
            )

    async def legacy_ready(gen_id: str) -> bool:
        return legacy_marker.exists() and (
            json.loads(legacy_marker.read_text(encoding="utf-8"))["generation"][
                "workerGenerationId"
            ]
            == gen_id
        )

    # Legacy driver without stopped-owner opt-in leaves seed file unchanged.
    legacy_gen_file = tmp_path / "legacy-generation.json"
    seed_bytes = json.dumps({"workerGenerationId": GEN1}).encode("utf-8")
    legacy_gen_file.write_bytes(seed_bytes)
    legacy_driver = SourceProcessDriver(
        scope=SCOPE,
        command=(str(Path(sys.executable)), "-I", str(legacy_child)),
        stop_file=legacy_stop,
        generation_file=legacy_gen_file,
        environment={},
        ready=legacy_ready,
        resume_camera_id=CAMERA_ID,
        resume_manifest_supported=True,
        resume_configuration_loader=load,
    )
    assert isinstance(legacy_driver, StoppedOwnerResumeDriver)
    assert not legacy_driver.supports_stopped_owner_resume(SCOPE)
    assert not legacy_driver.can_resume_stopped_owner(
        SCOPE,
        "0",
        ResumeBinding(cameraId=CAMERA_ID, workerGenerationId=GEN1, streamSessionId=SESS1),
    )
    await legacy_driver.start(SCOPE, "0", _generation(GEN2, SESS2))
    try:
        assert legacy_gen_file.read_bytes() == seed_bytes
    finally:
        await legacy_driver.stop(SCOPE)

    # Stopped-owner driver writes metadata-only generation.json on start and start_resume.
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child = state_dir / "child.py"
    marker = state_dir / "marker.json"
    stop_file = state_dir / "STOP"
    _write_stop_aware_child(child, marker, stop_file)
    gen_file = state_dir / "generation.json"
    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    assert driver.supports_stopped_owner_resume(SCOPE)
    gen1 = _generation(GEN1, SESS1, secret="super-secret-value-" + "x" * 30)
    await driver.start(SCOPE, "0", gen1)
    try:
        assert await driver.is_running(SCOPE, GEN1)
        written = json.loads(gen_file.read_text(encoding="utf-8"))
        assert written == {"workerGenerationId": GEN1}
        assert gen1["secret"] not in gen_file.read_text(encoding="utf-8")
        assert SESS1 not in gen_file.read_text(encoding="utf-8")
        assert [p for p in state_dir.iterdir() if p.name.endswith(".tmp")] == []
    finally:
        await driver.stop(SCOPE)

    gen2 = _generation(GEN2, SESS2, secret="another-secret-" + "y" * 30)
    manifest2 = ResumeManifest(
        generationBinding=ResumeBinding(
            cameraId=CAMERA_ID, workerGenerationId=GEN2, streamSessionId=SESS2
        ),
        configuration=CONFIG,
    )
    await driver.start_resume(SCOPE, "0", gen2, manifest2)
    try:
        assert await driver.is_running(SCOPE, GEN2)
        assert json.loads(gen_file.read_text(encoding="utf-8")) == {"workerGenerationId": GEN2}
        assert gen2["secret"] not in gen_file.read_text(encoding="utf-8")
    finally:
        await driver.stop(SCOPE)

    # Symlink generation.json blocks _start before spawning child.
    symlink_gen = tmp_path / "symlink-gen.json"
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"workerGenerationId": GEN1}), encoding="utf-8")
    try:
        symlink_gen.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    symlink_driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=symlink_gen,
        stop_file=state_dir / "STOP-symlink",
    )
    with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
        await symlink_driver.start(SCOPE, "0", _generation(GEN3, SESS3))
    assert symlink_driver.process is None


async def _always_false() -> bool:
    return False


@pytest.mark.anyio
async def test_stopped_owner_happy_path_after_parent_crash_with_expired_old_generation(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    child = tmp_path / "child.py"
    marker = tmp_path / "marker.json"
    gen_file = tmp_path / "generation.json"
    stop_file = tmp_path / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    key = SecretStr(Fernet.generate_key().decode())
    registry = SourceRegistry(tmp_path / "registry.enc", key)
    _seed_persisted_running(registry, gen_file)

    driver1 = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    client1, headers, base = _make_async_api(SourceActivator(registry, driver1))
    async with client1:
        # Dead parent must not claim RUNNING or synthesize a new generation on status.
        assert driver1.process is None
        assert (
            await client1.post(
                f"{base}/resume-status", headers=headers, json={"commandId": CMD_OLD}
            )
        ).status_code == 409
        assert (
            await client1.post(
                f"{base}/resume-status", headers=headers, json={"commandId": CMD_NEW}
            )
        ).status_code == 404

        # Prepare with expired old G1 succeeds into PREPARED without starting child.
        req_new = _prepare_request(
            command_id=CMD_NEW, worker_generation_id=GEN1, stream_session_id=SESS1
        )
        prepared = await client1.post(
            f"{base}/prepare-resume", headers=headers, json=req_new.model_dump()
        )
        assert prepared.status_code == 200
        assert prepared.json()["state"] == "PREPARED"
        assert prepared.json()["sourceRevision"] == 1
        assert driver1.process is None

    # Simulate another parent crash while command is PREPARED; retry succeeds if lease is free.
    driver_retry = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    client_retry, headers, base = _make_async_api(
        SourceActivator(SourceRegistry(tmp_path / "registry.enc", key), driver_retry)
    )
    async with client_retry:
        try:
            retried = await client_retry.post(
                f"{base}/prepare-resume", headers=headers, json=req_new.model_dump()
            )
            assert retried.status_code == 200 and retried.json()["state"] == "PREPARED"

            # Reject expired/reused generation in commit-resume before accepting fresh G2/S2.
            assert (
                await client_retry.post(
                    f"{base}/commit-resume",
                    headers=headers,
                    json={
                        "commandId": CMD_NEW,
                        "generation": _generation(GEN2, SESS2, expired=True),
                    },
                )
            ).status_code == 409
            assert (
                await client_retry.post(
                    f"{base}/commit-resume",
                    headers=headers,
                    json={"commandId": CMD_NEW, "generation": _generation(GEN1, SESS2)},
                )
            ).status_code == 409

            gen2 = _generation(GEN2, SESS2)
            committed = await client_retry.post(
                f"{base}/commit-resume",
                headers=headers,
                json={"commandId": CMD_NEW, "generation": gen2},
            )
            assert committed.status_code == 200
            assert committed.json()["state"] == "RUNNING"
            assert committed.json()["currentGenerationId"] == GEN2
            assert committed.json()["sourceRevision"] == 1
            assert json.loads(gen_file.read_text(encoding="utf-8")) == {"workerGenerationId": GEN2}
            assert driver_retry.owns_resume(
                SCOPE,
                "0",
                ResumeBinding(cameraId=CAMERA_ID, workerGenerationId=GEN2, streamSessionId=SESS2),
            )
        finally:
            await driver_retry.stop(SCOPE)

    # Second parent reboot -> stopped-owner recovery works again on G2 -> G3.
    driver2 = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    client2, headers, base = _make_async_api(
        SourceActivator(SourceRegistry(tmp_path / "registry.enc", key), driver2)
    )
    async with client2:
        try:
            req_3 = _prepare_request(
                command_id=CMD_3, worker_generation_id=GEN2, stream_session_id=SESS2
            )
            assert (
                await client2.post(
                    f"{base}/prepare-resume", headers=headers, json=req_3.model_dump()
                )
            ).status_code == 200
            gen3 = _generation(GEN3, SESS3)
            res3 = await client2.post(
                f"{base}/commit-resume",
                headers=headers,
                json={"commandId": CMD_3, "generation": gen3},
            )
            assert res3.status_code == 200 and res3.json()["currentGenerationId"] == GEN3
            assert json.loads(gen_file.read_text(encoding="utf-8")) == {"workerGenerationId": GEN3}
        finally:
            await driver2.stop(SCOPE)


@pytest.mark.anyio
async def test_stopped_owner_busy_os_lease_fails_closed_before_mutating_registry(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    child = tmp_path / "child.py"
    marker = tmp_path / "marker.json"
    gen_file = tmp_path / "generation.json"
    stop_file = tmp_path / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    key = SecretStr(Fernet.generate_key().decode())
    registry = SourceRegistry(tmp_path / "registry.enc", key)
    _seed_persisted_running(registry, gen_file)
    before_bytes = registry.path.read_bytes()

    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    client, headers, base = _make_async_api(SourceActivator(registry, driver))
    req = _prepare_request(command_id=CMD_NEW, worker_generation_id=GEN1, stream_session_id=SESS1)

    async with client:
        orphan_lease = ConfigurationProcessLease(lock_dir, RESOURCE_ID)
        orphan_lease.acquire()
        try:
            res = await client.post(
                f"{base}/prepare-resume", headers=headers, json=req.model_dump()
            )
            assert res.status_code == 409
            assert res.json()["code"] == "SOURCE_RECONCILIATION_REQUIRED"
            assert registry.path.read_bytes() == before_bytes
            assert registry.read(SCOPE)["state"] == "RUNNING"
        finally:
            orphan_lease.release()

        # Once orphan releases OS lease, prepare succeeds cleanly and still fails if retaken.
        assert (
            await client.post(f"{base}/prepare-resume", headers=headers, json=req.model_dump())
        ).status_code == 200
        assert (lock_dir / f"configuration-capture-{RESOURCE_ID}.lock").exists()

        external_lease = ConfigurationProcessLease(lock_dir, RESOURCE_ID)
        external_lease.acquire()
        try:
            retry = await client.post(
                f"{base}/prepare-resume", headers=headers, json=req.model_dump()
            )
            assert retry.status_code == 409
        finally:
            external_lease.release()


@pytest.mark.parametrize(
    "case",
    [
        "missing_gen_file",
        "corrupt_gen_json",
        "empty_gen_json",
        "duplicate_key_gen_json",
        "symlink_gen_file",
        "stale_gen_id",
        "state_stopping",
        "state_starting",
        "state_stopped",
        "wrong_stream_session",
        "wrong_generation_id",
        "wrong_revision",
        "wrong_camera_id",
        "replay_kind",
        "reconciliation_latched",
        "missing_prior_worker_id",
    ],
)
@pytest.mark.anyio
async def test_stopped_owner_uncertain_stale_missing_or_corrupt_state_fails_closed(
    tmp_path: Path, case: str
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    child = tmp_path / "child.py"
    marker = tmp_path / "marker.json"
    gen_file = tmp_path / "generation.json"
    stop_file = tmp_path / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    key = SecretStr(Fernet.generate_key().decode())
    registry = SourceRegistry(tmp_path / "registry.enc", key)
    _seed_persisted_running(
        registry,
        gen_file,
        state=(
            "STOPPING"
            if case == "state_stopping"
            else "STARTING"
            if case == "state_starting"
            else "STOPPED"
            if case == "state_stopped"
            else "RUNNING"
        ),
        source_kind="REPLAY" if case == "replay_kind" else "LIVE",
        include_gen_file=(case != "missing_gen_file"),
    )

    if case == "corrupt_gen_json":
        gen_file.write_bytes(b"{not-json")
    elif case == "empty_gen_json":
        gen_file.write_text("{}", encoding="utf-8")
    elif case == "duplicate_key_gen_json":
        gen_file.write_text(
            f'{{"workerGenerationId": "{GEN2}", "workerGenerationId": "{GEN1}"}}', encoding="utf-8"
        )
    elif case == "symlink_gen_file":
        gen_file.unlink()
        target = tmp_path / "symlink-target.json"
        target.write_text(json.dumps({"workerGenerationId": GEN1}), encoding="utf-8")
        try:
            gen_file.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Symlink creation unavailable: {exc}")
    elif case == "stale_gen_id":
        gen_file.write_text(json.dumps({"workerGenerationId": GEN2}), encoding="utf-8")
    elif case == "missing_prior_worker_id":
        row = registry.read(SCOPE)
        assert row is not None
        row["generation"] = {
            "streamSessionId": SESS1,
            "secret": "s" * 43,
            "expiresAt": "2099-01-01T00:00:00Z",
        }
        registry.write(SCOPE, row)

    before_bytes = registry.path.read_bytes()
    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    if case == "reconciliation_latched":
        driver._reconciliation_required = True

    req = _prepare_request(
        command_id=CMD_NEW,
        worker_generation_id=GEN2 if case == "wrong_generation_id" else GEN1,
        stream_session_id=SESS2 if case == "wrong_stream_session" else SESS1,
        expected_revision=2 if case == "wrong_revision" else 1,
        camera_id=str(uuid4()) if case == "wrong_camera_id" else CAMERA_ID,
    )
    client, headers, base = _make_async_api(SourceActivator(registry, driver))
    async with client:
        response = await client.post(
            f"{base}/prepare-resume", headers=headers, json=req.model_dump()
        )
    assert response.status_code == 409
    assert response.json()["code"] == "SOURCE_RECONCILIATION_REQUIRED"
    assert registry.path.read_bytes() == before_bytes
    assert driver.process is None


@pytest.mark.anyio
async def test_stopped_owner_can_resume_initial_capture_running_row_without_manifest(
    tmp_path: Path,
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    child = tmp_path / "child.py"
    marker = tmp_path / "marker.json"
    gen_file = tmp_path / "generation.json"
    stop_file = tmp_path / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    key = SecretStr(Fernet.generate_key().decode())
    registry = SourceRegistry(tmp_path / "registry.enc", key)
    _seed_persisted_running(registry, gen_file, with_resume_manifest=False)

    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    client, headers, base = _make_async_api(SourceActivator(registry, driver))
    req = _prepare_request(command_id=CMD_NEW, worker_generation_id=GEN1, stream_session_id=SESS1)
    async with client:
        try:
            assert (
                await client.post(f"{base}/prepare-resume", headers=headers, json=req.model_dump())
            ).status_code == 200
            gen2 = _generation(GEN2, SESS2)
            res = await client.post(
                f"{base}/commit-resume",
                headers=headers,
                json={"commandId": CMD_NEW, "generation": gen2},
            )
            assert res.status_code == 200 and res.json()["currentGenerationId"] == GEN2
        finally:
            await driver.stop(SCOPE)


@pytest.mark.anyio
async def test_operator_app_wires_stopped_owner_recovery_without_boot_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path, doc, environ = _operator_bundle(tmp_path)
    state_dir = Path(str(doc["stateDirectory"]))
    lock_dir = state_dir / "locks"
    site_id = str(doc["siteId"])
    camera_id = str(doc["cameraId"])
    camera_ext = str(doc["cameraExternalId"])
    scope = (site_id, camera_ext)

    valid_bundle = policy_bundle(outside=True)
    assert valid_bundle.camera_id == camera_id
    bundle_cfg = ResumeConfiguration(
        configurationVersion=valid_bundle.configuration_version,
        viewRevision=valid_bundle.view_revision,
        bundleSha256=valid_bundle.bundle_sha256,
    )

    child = tmp_path / "fake_operator_child.py"
    marker = tmp_path / "fake_child_marker.json"
    stop_file = state_dir / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    # Seed persisted RUNNING registry + generation.json matching expired G1 as after reboot.
    registry = SourceRegistry(
        state_dir / "source-registry.enc",
        SecretStr(environ["SMARTSITE_AI_SOURCE_REGISTRY_KEY"]),
    )
    registry.write(
        scope,
        {
            "state": "RUNNING",
            "commandId": CMD_OLD,
            "requestHash": "f" * 64,
            "sourceRevision": 1,
            "sourceKind": "LIVE",
            "source": "0",
            "currentGenerationId": GEN1,
            "generation": _generation(GEN1, SESS1, expired=True),
        },
    )
    gen_file = state_dir / "generation.json"
    gen_file.write_text(json.dumps({"workerGenerationId": GEN1}), encoding="utf-8")

    # Mock Backend bundle fetch & preview image probe offline.
    real_client = op.BackendConfigurationClient

    def fake_backend_client(
        url: str, token: SecretStr, **kwargs: object
    ) -> op.BackendConfigurationClient:
        return real_client(
            url,
            token,
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(
                    200,
                    headers={"ETag": f'"sha256:{valid_bundle.bundle_sha256}"'},
                    json=valid_bundle.model_dump(mode="json", by_alias=True),
                )
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    async def fake_probe(
        _origin: str, _token: SecretStr, ext_id: str, generation_id: str, **_kwargs: object
    ) -> bool:
        if not marker.exists() or ext_id != camera_ext:
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload["generation"]["workerGenerationId"] == generation_id and bool(
            _valid_image_payload(camera_ext)
        )

    monkeypatch.setattr(op, "BackendConfigurationClient", fake_backend_client)
    monkeypatch.setattr(op, "probe_owned_image", fake_probe)

    app, driver = op.build_operator_app(profile_path, environ)
    # Boot never spawns child or starts camera automatically.
    assert driver.process is None
    assert driver.current_source is None
    assert driver.capture_lease_directory == lock_dir
    assert driver.capture_lease_resource_id == RESOURCE_ID
    assert driver.supports_stopped_owner_resume(scope)

    # Override child command after verifying build_operator_app wired capture lease cleanly.
    driver.command = (str(Path(sys.executable)), "-I", str(child))
    driver.resume_command = driver.command

    headers = {"Authorization": f"Bearer {environ['SMARTSITE_AI_SOURCE_CONTROL_TOKEN']}"}
    base = f"/v1/source-control/{site_id}/{camera_ext}"
    req = ResumePrepare(
        commandId=CMD_NEW,
        expectedRevision=1,
        generationBinding=ResumeBinding(
            cameraId=camera_id, workerGenerationId=GEN1, streamSessionId=SESS1
        ),
        configuration=bundle_cfg,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        try:
            assert (
                await client.post(f"{base}/prepare-resume", headers=headers, json=req.model_dump())
            ).status_code == 200
            assert driver.process is None

            gen2 = _generation(GEN2, SESS2)
            commit_res = await client.post(
                f"{base}/commit-resume",
                headers=headers,
                json={"commandId": CMD_NEW, "generation": gen2},
            )
            assert commit_res.status_code == 200
            assert commit_res.json()["currentGenerationId"] == GEN2
            assert json.loads(gen_file.read_text(encoding="utf-8")) == {"workerGenerationId": GEN2}
            assert gen2["secret"] not in gen_file.read_text(encoding="utf-8")
            assert driver.owns_resume(
                scope,
                "0",
                ResumeBinding(cameraId=camera_id, workerGenerationId=GEN2, streamSessionId=SESS2),
            )
        finally:
            await driver.stop(scope)


def test_legacy_fake_driver_without_stopped_owner_protocol_still_rejects(tmp_path: Path) -> None:
    import asyncio

    registry = SourceRegistry(
        tmp_path / "legacy-fake.enc", SecretStr(Fernet.generate_key().decode())
    )
    registry.write(
        LEGACY_SCOPE,
        {
            "state": "RUNNING",
            "commandId": str(uuid4()),
            "sourceRevision": 3,
            "sourceKind": "LIVE",
            "source": "0",
            "currentGenerationId": LEGACY_OLD["workerGenerationId"],
            "generation": LEGACY_OLD,
        },
    )
    driver = LegacyResumeDriver()
    driver.owned = False
    assert not isinstance(driver, StoppedOwnerResumeDriver)
    activator = SourceActivator(registry, driver)
    with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
        asyncio.run(activator.prepare_resume(LEGACY_SCOPE, legacy_request()))


@pytest.mark.anyio
async def test_stopped_owner_zero_progress_write_fault_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    child = state_dir / "child.py"
    marker = state_dir / "marker.json"
    stop_file = state_dir / "STOP"
    _write_stop_aware_child(child, marker, stop_file)

    gen_file = state_dir / "generation.json"
    original_bytes = json.dumps({"workerGenerationId": GEN1}, separators=(",", ":")).encode("utf-8")
    gen_file.write_bytes(original_bytes)

    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )

    real_open = os.open
    real_write = os.write
    captured_fds: set[int] = set()
    zero_write_calls = 0

    def tracking_open(path: object, flags: int, mode: int = 0o777, *args: object) -> int:
        fd = real_open(path, flags, mode, *args)  # type: ignore[arg-type]
        if Path(os.fspath(path)).parent == state_dir.resolve():  # type: ignore[arg-type]
            captured_fds.add(fd)
        return fd

    def zero_progress_write(fd: int, buf: bytes) -> int:
        nonlocal zero_write_calls
        if fd in captured_fds:
            zero_write_calls += 1
            return 0
        return real_write(fd, buf)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "write", zero_progress_write)

    with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
        await driver.start(SCOPE, "0", _generation(GEN2, SESS2))

    assert zero_write_calls == 1
    assert driver.process is None
    assert gen_file.read_bytes() == original_bytes
    assert await driver.generation_id(SCOPE) == GEN1
    assert [p for p in state_dir.iterdir() if p.name.endswith(".tmp")] == []


@pytest.mark.anyio
async def test_generation_id_bounds_read_and_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    child = tmp_path / "child.py"
    marker = tmp_path / "marker.json"
    stop_file = tmp_path / "STOP"
    _write_stop_aware_child(child, marker, stop_file)
    gen_file = tmp_path / "generation.json"

    oversized_text = json.dumps({"workerGenerationId": GEN1}) + (" " * 17_000)
    gen_file.write_text(oversized_text, encoding="utf-8")

    read_limits: list[int | None] = []
    real_open = Path.open

    class _RecordingReader:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __enter__(self) -> _RecordingReader:
            self._inner.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self._inner.__exit__(*args)  # type: ignore[attr-defined]

        def read(self, size: int = -1) -> bytes:
            read_limits.append(size)
            return self._inner.read(size)  # type: ignore[attr-defined]

    def spy_open(self: Path, *args: object, **kwargs: object) -> object:
        handle = real_open(self, *args, **kwargs)  # type: ignore[arg-type]
        if self == gen_file:
            return _RecordingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", spy_open)

    driver = _driver(
        child_script=child,
        marker=marker,
        lock_dir=lock_dir,
        generation_file=gen_file,
        stop_file=stop_file,
    )
    with pytest.raises(SourceActivationError, match="SOURCE_RECONCILIATION_REQUIRED"):
        await driver.generation_id(SCOPE)
    assert read_limits == [16_385]
