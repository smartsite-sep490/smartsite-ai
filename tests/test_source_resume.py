import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from smartsite_ai.source_activation import SourceActivationError, SourceActivator, SourceRegistry
from smartsite_ai.source_resume_models import ResumeConfiguration, ResumeManifest, ResumePrepare

SCOPE = ("11111111-1111-4111-8111-111111111111", "camera-a")
CAMERA = str(uuid4())
OLD = {
    "workerGenerationId": str(uuid4()),
    "streamSessionId": str(uuid4()),
    "secret": "x" * 48,
    "expiresAt": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
}


def request():
    return ResumePrepare(
        commandId=str(uuid4()),
        expectedRevision=3,
        generationBinding={
            "cameraId": CAMERA,
            "workerGenerationId": OLD["workerGenerationId"],
            "streamSessionId": OLD["streamSessionId"],
        },
        configuration={"configurationVersion": 12, "viewRevision": 7, "bundleSha256": "a" * 64},
    )


class Driver:
    def __init__(self):
        self.calls = []
        self.owned = True
        self.session = OLD["streamSessionId"]
        self.running = False

    def supports(self, scope):
        return scope == SCOPE

    def supports_resume(self, scope):
        return scope == SCOPE

    def owns_resume(self, scope, source, binding):
        return self.owned and source == "0" and binding.streamSessionId == self.session

    async def generation_id(self, scope):
        return OLD["workerGenerationId"]

    async def stop(self, scope):
        self.calls.append("stop")
        self.owned = False

    async def start_resume(self, scope, source, generation, manifest):
        self.calls.append(manifest)
        self.running = True
        self.owned = True
        self.session = generation["streamSessionId"]

    async def is_running(self, scope, generation):
        return self.running


def setup(tmp_path):
    driver = Driver()
    registry = SourceRegistry(tmp_path / "state.enc", SecretStr(Fernet.generate_key().decode()))
    registry.write(
        SCOPE,
        {
            "state": "RUNNING",
            "commandId": str(uuid4()),
            "sourceRevision": 3,
            "sourceKind": "LIVE",
            "source": "0",
            "currentGenerationId": OLD["workerGenerationId"],
            "generation": OLD,
        },
    )
    return SourceActivator(registry, driver), driver


def test_resume_saved_source_preserves_revision_and_is_idempotent(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        prepared = await active.prepare_resume(SCOPE, req)
        assert prepared["state"] == "PREPARED" and prepared["sourceRevision"] == 3
        assert await active.prepare_resume(SCOPE, req) == prepared
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        result = await active.commit_resume(SCOPE, req.commandId, new)
        assert (
            result["sourceRevision"] == 3
            and result["currentGenerationId"] == new["workerGenerationId"]
        )
        assert await active.prepare_resume(SCOPE, req) == result
        assert await active.commit_resume(SCOPE, req.commandId, new) == result
        assert len(driver.calls) == 2 and isinstance(driver.calls[1], ResumeManifest)
        assert driver.calls[1].generationBinding.streamSessionId == new["streamSessionId"]
        assert driver.calls[1].configuration == req.configuration
        with pytest.raises(SourceActivationError):
            await active.prepare_resume(SCOPE, req.model_copy(update={"expectedRevision": 4}))
        with pytest.raises(SourceActivationError):
            await active.commit(SCOPE, req.commandId, new)
        driver.running = False
        with pytest.raises(SourceActivationError, match="RECONCILIATION"):
            await active.prepare_resume(SCOPE, req)

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["unknown_owner", "foreign_session", "stale_source", "replay"])
def test_resume_rejects_without_stopping(tmp_path, changed):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        if changed == "unknown_owner":
            driver.owned = False
        if changed == "foreign_session":
            req = req.model_copy(
                update={
                    "generationBinding": req.generationBinding.model_copy(
                        update={"streamSessionId": str(uuid4())}
                    )
                }
            )
        if changed == "stale_source":
            req = req.model_copy(update={"expectedRevision": 2})
        if changed == "replay":
            row = active.registry.read(SCOPE)
            row["sourceKind"] = "REPLAY"
            active.registry.write(SCOPE, row)
        with pytest.raises(SourceActivationError):
            await active.prepare_resume(SCOPE, req)
        assert driver.calls == []

    asyncio.run(run())


def test_resume_uncertain_stop_cannot_be_retried(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()

        async def cancelled(scope):
            raise asyncio.CancelledError()

        driver.stop = cancelled
        with pytest.raises(asyncio.CancelledError):
            await active.prepare_resume(SCOPE, req)
        assert active.registry.read(SCOPE)["state"] == "STOPPING"
        with pytest.raises(SourceActivationError, match="RECONCILIATION"):
            await active.prepare_resume(SCOPE, req)

    asyncio.run(run())


def test_typed_configuration_refuses_missing_extra_and_unsafe_numbers():
    for data in [
        {},
        {"configurationVersion": True, "viewRevision": 1, "bundleSha256": "a" * 64},
        {"configurationVersion": 2**53, "viewRevision": 1, "bundleSha256": "a" * 64},
    ]:
        with pytest.raises(ValueError):
            ResumeConfiguration.model_validate(data)


def test_process_driver_default_does_not_support_resume(tmp_path):
    import sys
    from pathlib import Path

    from smartsite_ai.source_process_driver import SourceProcessDriver

    async def ready(_):
        return True

    driver = SourceProcessDriver(
        scope=SCOPE,
        command=(str(Path(sys.executable)),),
        stop_file=tmp_path / "stop",
        generation_file=tmp_path / "gen",
        environment={},
        ready=ready,
    )
    assert not driver.supports_resume(SCOPE)
    assert not driver.owns_resume(SCOPE, "0", request().generationBinding)

    async def load(manifest, generation):
        return request().configuration

    unguarded = SourceProcessDriver(
        scope=SCOPE,
        command=(str(Path(sys.executable)),),
        stop_file=tmp_path / "stop",
        generation_file=tmp_path / "gen",
        environment={},
        ready=ready,
        resume_camera_id=CAMERA,
        resume_configuration_loader=load,
    )
    assert not unguarded.supports_resume(SCOPE)


def test_guarded_driver_reaps_known_exited_owner_and_transmits_exact_manifest(tmp_path):
    import json
    import sys
    from pathlib import Path

    from smartsite_ai.source_process_driver import SourceProcessDriver

    async def run():
        child = tmp_path / "child.py"
        output = tmp_path / "manifest.json"
        child.write_text(
            "import os,pathlib,time\n"
            f"out=pathlib.Path({str(output)!r})\n"
            'manifest=os.environ.get("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION")\n'
            "if manifest: out.write_text(manifest)\n"
            "else: raise SystemExit(7)\n"
            "time.sleep(30)\n"
        )

        async def ready(_):
            return output.exists()

        loaded = ResumeConfiguration(configurationVersion=12, viewRevision=7, bundleSha256="a" * 64)

        async def load(manifest, generation):
            return loaded

        driver = SourceProcessDriver(
            scope=SCOPE,
            command=(str(Path(sys.executable)), "-I", str(child)),
            stop_file=tmp_path / "stop",
            generation_file=tmp_path / "gen",
            environment={},
            ready=ready,
            resume_camera_id=CAMERA,
            resume_manifest_supported=True,
            resume_configuration_loader=load,
        )
        assert not driver.owns_resume(SCOPE, "0", request().generationBinding)
        await driver.start(SCOPE, "0", OLD)
        await driver.process.wait()
        assert driver.current_source is None
        assert driver.owns_resume(SCOPE, "0", request().generationBinding)
        await driver.stop(SCOPE)
        assert not driver.owns_resume(SCOPE, "0", request().generationBinding)
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        manifest = ResumeManifest(
            generationBinding={
                "cameraId": CAMERA,
                "workerGenerationId": new["workerGenerationId"],
                "streamSessionId": new["streamSessionId"],
            },
            configuration=loaded,
        )
        wrong = manifest.model_copy(
            update={"configuration": loaded.model_copy(update={"configurationVersion": 13})}
        )
        with pytest.raises(ValueError, match="configuration changed"):
            await driver.start_resume(SCOPE, "0", new, wrong)
        assert driver.process is None
        with pytest.raises(ValueError):
            await driver.start_resume(SCOPE, "0", new, None)
        assert driver.process is None
        await driver.start_resume(SCOPE, "0", new, manifest)
        try:
            assert await driver.is_running(SCOPE, new["workerGenerationId"])
            assert not await driver.is_running(SCOPE, OLD["workerGenerationId"])
            assert json.loads(output.read_text()) == manifest.model_dump()
        finally:
            driver.process.kill()
            await driver.stop(SCOPE)

    asyncio.run(run())


def test_legacy_driver_cannot_prepare_resume(tmp_path):
    from test_source_activation import Driver as LegacyDriver

    active, _ = setup(tmp_path)
    active.driver = LegacyDriver()
    before = active.registry.read(SCOPE)
    with pytest.raises(SourceActivationError, match="UNAVAILABLE"):
        asyncio.run(active.prepare_resume(SCOPE, request()))
    assert active.registry.read(SCOPE) == before


def test_failed_start_persists_uncertainty_and_never_restarts_on_retry(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        await active.prepare_resume(SCOPE, req)

        async def fail(scope, source, generation, manifest):
            driver.calls.append("failed-start")
            raise RuntimeError("Synthetic startup failure")

        driver.start_resume = fail
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        with pytest.raises(RuntimeError, match="Synthetic"):
            await active.commit_resume(SCOPE, req.commandId, new)
        assert active.registry.read(SCOPE)["state"] == "STARTING"
        with pytest.raises(SourceActivationError):
            await active.commit_resume(SCOPE, req.commandId, new)
        with pytest.raises(SourceActivationError):
            await active.commit(SCOPE, req.commandId, new)
        assert driver.calls == ["stop", "failed-start"]

    asyncio.run(run())


def test_prepare_running_retry_requires_exact_owned_new_session(tmp_path):
    async def run():
        active, driver = setup(tmp_path)
        req = request()
        await active.prepare_resume(SCOPE, req)
        new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
        await active.commit_resume(SCOPE, req.commandId, new)
        driver.owned = False
        with pytest.raises(SourceActivationError, match="RECONCILIATION"):
            await active.prepare_resume(SCOPE, req)

    asyncio.run(run())


def test_resume_api_uses_scoped_auth_strict_body_and_default_unavailable(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from smartsite_ai.source_control_api import source_control_router
    from smartsite_ai.source_control_models import SourcePolicy

    active, driver = setup(tmp_path)

    async def probe(*args):
        raise AssertionError("Resume never probes a source")

    app = FastAPI()
    app.include_router(
        source_control_router(
            host_id="edge",
            label="Host",
            scopes={SCOPE},
            policy=SourcePolicy({}, {}, ()),
            service_token=SecretStr("a" * 48),
            probe=probe,
            activator=active,
        )
    )
    client = TestClient(app)
    url = f"/v1/source-control/{SCOPE[0]}/{SCOPE[1]}"
    headers = {"Authorization": "Bearer " + "a" * 48}
    req = request()
    assert (
        client.get(url + "/capabilities", headers=headers).json()["configurationResumeSupported"]
        is True
    )
    assert client.post(url + "/prepare-resume", json=req.model_dump()).status_code == 401
    assert (
        client.post(
            url + "/prepare-resume", headers=headers, json={**req.model_dump(), "source": "evil"}
        ).status_code
        == 400
    )
    assert (
        client.post(url + "/prepare-resume", headers=headers, json=req.model_dump()).json()[
            "sourceRevision"
        ]
        == 3
    )
    new = {**OLD, "workerGenerationId": str(uuid4()), "streamSessionId": str(uuid4())}
    result = client.post(
        url + "/commit-resume",
        headers=headers,
        json={"commandId": req.commandId, "generation": new},
    )
    assert result.status_code == 200 and result.json()["state"] == "RUNNING"
    assert "secret" not in result.text and "source" not in result.json()
    from test_source_activation import Driver as LegacyDriver

    active.driver = LegacyDriver()
    assert (
        "configurationResumeSupported"
        not in client.get(url + "/capabilities", headers=headers).json()
    )
    assert (
        client.post(url + "/prepare-resume", headers=headers, json=req.model_dump()).status_code
        == 409
    )
    assert (
        client.post(
            url.replace("camera-a", "foreign") + "/prepare-resume",
            headers=headers,
            json=req.model_dump(),
        ).status_code
        == 404
    )


@pytest.mark.parametrize(
    "changed", ["camera", "generation", "session", "version", "view", "hash", "missing", "match"]
)
def test_checked_in_runtime_consumes_manifest_before_model_or_capture(
    tmp_path, monkeypatch, changed
):
    import json

    import httpx
    from test_tracked_runtime_selection import document

    from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
    from smartsite_ai.tools import run_camera_runtime as runtime

    value, bundle = document(tmp_path)
    manifest_path = tmp_path / "runtime.json"
    manifest_path.write_text(json.dumps(value))
    credential = json.loads((tmp_path / "generation.json").read_text())
    expected = {
        "generationBinding": {
            key: credential[key] for key in ("cameraId", "workerGenerationId", "streamSessionId")
        },
        "configuration": {
            "configurationVersion": bundle.configuration_version,
            "viewRevision": bundle.view_revision,
            "bundleSha256": bundle.bundle_sha256,
        },
    }
    if changed in {"camera", "generation", "session"}:
        key = {
            "camera": "cameraId",
            "generation": "workerGenerationId",
            "session": "streamSessionId",
        }[changed]
        expected["generationBinding"][key] = str(uuid4())
    if changed in {"version", "view", "hash"}:
        key = {"version": "configurationVersion", "view": "viewRevision", "hash": "bundleSha256"}[
            changed
        ]
        expected["configuration"][key] = "c" * 64 if changed == "hash" else 99
    if changed == "missing":
        monkeypatch.delenv("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", raising=False)
    else:
        monkeypatch.setenv(
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", json.dumps(expected)
        )

    def response(request):
        return httpx.Response(
            200,
            json=bundle.model_dump(mode="json", by_alias=True),
            headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
        )

    def client(*args, **kwargs):
        return BackendConfigurationClient(
            "https://backend.example",
            "synthetic-service",
            policy_v2=True,
            transport=httpx.MockTransport(response),
        )

    monkeypatch.setattr(BackendConfigurationClient, "from_settings", client)
    calls = []

    def model(*args, **kwargs):
        calls.append("model")
        raise RuntimeError("Synthetic model boundary reached")

    monkeypatch.setattr(runtime, "load_yolo11_detector", model)
    with pytest.raises((ValueError, RuntimeError)) as error:
        asyncio.run(runtime.run_manifest(manifest_path, require_resume_manifest=True))
    assert calls == (["model"] if changed == "match" else [])
    if changed == "match":
        assert "Synthetic model boundary" in str(error.value)
