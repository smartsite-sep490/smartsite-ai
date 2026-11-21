import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from test_source_control_child_profile import inputs
from test_tracked_runtime_selection import document

from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
from smartsite_ai.integrations.outbox import SqliteEventOutbox
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.runtime.assembly import close_configuration_clients, preflight_manifest
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest
from smartsite_ai.runtime.source_control_child import GENERATION_ENV, SOURCE_ENV
from smartsite_ai.source_control_models import SourcePolicy


def _with_fresh_secret(env: dict[str, str], secret: str) -> dict[str, str]:
    updated = dict(env)
    generation = json.loads(updated[GENERATION_ENV])
    generation["secret"] = secret
    updated[GENERATION_ENV] = json.dumps(generation)
    return updated


def _bundle_client(bundle, requests: list[httpx.Request]) -> BackendConfigurationClient:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=bundle.model_dump(mode="json", by_alias=True),
            headers={"ETag": f'"sha256:{bundle.bundle_sha256}"'},
        )

    return BackendConfigurationClient(
        "https://backend.example",
        "synthetic-service",
        policy_v2=True,
        transport=httpx.MockTransport(respond),
    )


def test_memory_child_binding_returns_bound_manifest_and_secret_without_writing_file(
    tmp_path, monkeypatch
):
    import smartsite_ai.runtime.source_control_child as child
    from smartsite_ai.runtime.source_control_child import bind_source_control_child_memory

    def never_publish(*args):
        pytest.fail("Memory-only binding must never publish a credential file")

    monkeypatch.setattr(child, "_persist_credential", never_publish)
    value, bundle, path, original, expected, env = inputs(tmp_path)
    fresh_secret = "z" * 43
    env = _with_fresh_secret(env, fresh_secret)
    env[SOURCE_ENV] = "rtsp://operator:private-password@192.168.1.50/live"

    manifest = parse_manifest(value)
    bound, credential = bind_source_control_child_memory(
        manifest,
        policy=SourcePolicy({}, {}, ("192.168.1.0/24",)),
        scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    assert isinstance(credential, WorkerGenerationCredential)
    assert isinstance(credential.secret, SecretStr)
    assert credential.secret.get_secret_value() == fresh_secret
    assert credential.camera_id == expected.generationBinding.cameraId
    assert credential.worker_generation_id == expected.generationBinding.workerGenerationId
    assert credential.stream_session_id == expected.generationBinding.streamSessionId
    assert bound.cameras[0].source.kind == "env"
    assert bound.cameras[0].source.name == SOURCE_ENV
    assert bound.cameras[0].generation_credential_file == str(path)
    assert bound.cameras[0].outbox == manifest.cameras[0].outbox
    assert "private-password" not in bound.model_dump_json()
    assert fresh_secret not in bound.model_dump_json()
    assert fresh_secret not in repr(credential)
    assert fresh_secret not in credential.model_dump_json()


@pytest.mark.parametrize(
    "invalid",
    [
        "expired",
        "camera",
        "binding",
        "session",
        "scope",
        "source",
        "video",
        "replay",
        "untracked",
        "secret",
        "partial",
        "extra-credential-field",
    ],
)
def test_memory_child_binding_rejects_invalid_inputs_without_leak_or_write(
    tmp_path, invalid, monkeypatch
):
    import smartsite_ai.runtime.source_control_child as child
    from smartsite_ai.runtime.source_control_child import bind_source_control_child_memory

    def never_publish(*args):
        pytest.fail("Rejected memory binding must never publish a credential file")

    monkeypatch.setattr(child, "_persist_credential", never_publish)
    value, bundle, path, original, expected, env = inputs(tmp_path)
    fresh_secret = "z" * 43
    env = _with_fresh_secret(env, fresh_secret)
    camera = UUID(bundle.camera_id)
    scopes = {(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)}

    if invalid == "camera":
        camera = uuid4()
    elif invalid == "binding":
        expected = expected.model_copy(
            update={
                "generationBinding": expected.generationBinding.model_copy(
                    update={"workerGenerationId": str(uuid4())}
                )
            }
        )
    elif invalid == "session":
        expected = expected.model_copy(
            update={
                "generationBinding": expected.generationBinding.model_copy(
                    update={"streamSessionId": str(uuid4())}
                )
            }
        )
    elif invalid == "scope":
        scopes = set()
    elif invalid == "source":
        env[SOURCE_ENV] = "rtsp://169.254.169.254/live"
    elif invalid == "video":
        env[SOURCE_ENV] = str(tmp_path / "clip.mp4")
    elif invalid == "replay":
        value["cameras"][0]["live"] = False
    elif invalid == "untracked":
        value["cameras"][0].pop("trackedV17")
    elif invalid == "secret":
        data = json.loads(env[GENERATION_ENV])
        data["secret"] = "bad-short-secret"
        env[GENERATION_ENV] = json.dumps(data)
    elif invalid == "partial":
        env.pop(SOURCE_ENV)
    else:
        data = json.loads(env[GENERATION_ENV])
        if invalid == "extra-credential-field":
            data["credentialPath"] = str(tmp_path / "arbitrary.json")
        else:
            data["expiresAt"] = "2020-01-01T00:00:00.000Z"
        env[GENERATION_ENV] = json.dumps(data)

    with pytest.raises(CameraRuntimeError) as exc_info:
        bind_source_control_child_memory(
            parse_manifest(value),
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            scopes=scopes,
            camera_id=camera,
            environ=env,
            expected=expected,
        )
    assert fresh_secret not in str(exc_info.value)
    assert fresh_secret not in repr(exc_info.value)
    assert path.read_bytes() == original


def test_preflight_uses_default_file_without_override_and_revalidates_memory_override(tmp_path):
    value, bundle = document(tmp_path)
    bound = bind_manifest(parse_manifest(value), {})
    requests: list[httpx.Request] = []

    def legacy():
        pytest.fail("Tracked v1.7 preflight must not fetch legacy configuration")

    def v2():
        return _bundle_client(bundle, requests)

    async def exercise():
        failed_default, plans_default = await preflight_manifest(
            bound, configuration_client_factory=legacy, policy_v2_client_factory=v2
        )
        try:
            assert failed_default is None and len(plans_default) == 1
            assert (
                plans_default[0].generation.worker_generation_id
                == "00000000-0000-4000-8000-000000000031"
            )
            assert plans_default[0].generation.secret.get_secret_value() == "x" * 43
        finally:
            await close_configuration_clients(plans_default)

        override = WorkerGenerationCredential.model_validate(
            {
                "schemaVersion": "1.0.0",
                "cameraId": bundle.camera_id,
                "workerGenerationId": str(uuid4()),
                "streamSessionId": str(uuid4()),
                "secret": "z" * 43,
                "expiresAt": (datetime.now(UTC) + timedelta(minutes=10))
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            }
        )
        failed_override, plans_override = await preflight_manifest(
            bound,
            configuration_client_factory=legacy,
            policy_v2_client_factory=v2,
            generation_override=override,
        )
        try:
            assert failed_override is None and len(plans_override) == 1
            assert plans_override[0].generation == override
            assert plans_override[0].generation.secret.get_secret_value() == "z" * 43
        finally:
            await close_configuration_clients(plans_override)

    asyncio.run(exercise())
    assert len(requests) == 2


@pytest.mark.parametrize("case", ["wrong-camera", "expired", "untracked", "multi-camera", "replay"])
def test_preflight_rejects_invalid_generation_override_before_configuration(tmp_path, case):
    value, bundle = document(tmp_path)
    override_camera = str(uuid4()) if case == "wrong-camera" else bundle.camera_id
    expires = (
        "2020-01-01T00:00:00.000Z"
        if case == "expired"
        else (datetime.now(UTC) + timedelta(minutes=10))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    override = WorkerGenerationCredential.model_validate(
        {
            "schemaVersion": "1.0.0",
            "cameraId": override_camera,
            "workerGenerationId": str(uuid4()),
            "streamSessionId": str(uuid4()),
            "secret": "z" * 43,
            "expiresAt": expires,
        }
    )

    if case == "untracked":
        value["cameras"][0].pop("trackedV17")
    elif case == "replay":
        value["cameras"][0].pop("trackedV17")
        value["cameras"][0]["live"] = False
    elif case == "multi-camera":
        second = json.loads(json.dumps(value["cameras"][0]))
        second_cred = tmp_path / "generation2.json"
        second_cred.write_text((tmp_path / "generation.json").read_text())
        second["streamId"] = "synthetic2"
        second["cameraId"] = str(uuid4())
        second["cameraExternalId"] = "cam-2"
        second["source"] = {"kind": "path", "path": "1"}
        second["outbox"] = str(tmp_path / "outbox2.sqlite")
        second["evidenceDir"] = str(tmp_path / "ev2")
        second["generationCredentialFile"] = str(second_cred)
        (tmp_path / "ev2").mkdir()
        value["cameras"].append(second)

    bound = bind_manifest(parse_manifest(value), {})
    factory_calls: list[str] = []
    requests: list[httpx.Request] = []

    def client():
        factory_calls.append("called")
        return _bundle_client(bundle, requests)

    async def exercise():
        failure, plans = await preflight_manifest(
            bound,
            configuration_client_factory=client,
            policy_v2_client_factory=client,
            generation_override=override,
        )
        assert failure is not None
        assert plans == ()

    asyncio.run(exercise())
    assert factory_calls == []
    assert requests == []


def _setup_runtime_files(tmp_path):
    value, bundle, cred_path, original, expected, env = inputs(tmp_path)
    fresh_secret = "z" * 43
    env = _with_fresh_secret(env, fresh_secret)
    manifest_path = tmp_path / "runtime.json"
    manifest_path.write_text(json.dumps(value))
    profile_path = tmp_path / "host.json"
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
        )
    )
    return value, bundle, cred_path, original, expected, env, manifest_path, profile_path


@pytest.mark.skipif(os.name != "nt", reason="Windows memory-only runtime child binding")
def test_windows_run_manifest_uses_memory_generation_consumes_env_and_preserves_disk_file(
    tmp_path, monkeypatch
):
    from smartsite_ai.tools import run_camera_runtime as runtime

    value, bundle, cred_path, original, expected, env, manifest_path, profile_path = (
        _setup_runtime_files(tmp_path)
    )
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(
        "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json()
    )

    requests: list[httpx.Request] = []
    monkeypatch.setattr(
        BackendConfigurationClient,
        "from_settings",
        lambda *args, **kwargs: _bundle_client(bundle, requests),
    )

    observed_overrides: list[WorkerGenerationCredential | None] = []
    real_preflight = runtime.preflight_manifest

    async def spy_preflight(*args, **kwargs):
        observed_overrides.append(kwargs.get("generation_override"))
        return await real_preflight(*args, **kwargs)

    monkeypatch.setattr(runtime, "preflight_manifest", spy_preflight)

    model_calls: list[str] = []

    def fake_model(*args, **kwargs):
        model_calls.append("model")
        raise RuntimeError("Synthetic model boundary")

    monkeypatch.setattr(runtime, "load_yolo11_detector", fake_model)

    with pytest.raises(RuntimeError, match="Synthetic model boundary"):
        asyncio.run(
            runtime.run_manifest(
                manifest_path,
                source_control_profile=profile_path,
                expected_camera_id=UUID(bundle.camera_id),
            )
        )

    assert model_calls == ["model"]
    assert len(requests) == 1
    assert GENERATION_ENV not in os.environ
    assert cred_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    assert len(observed_overrides) == 1 and observed_overrides[0] is not None
    assert (
        observed_overrides[0].worker_generation_id == expected.generationBinding.workerGenerationId
    )
    assert observed_overrides[0].secret.get_secret_value() == "z" * 43
    owner = SqliteEventOutbox(tmp_path / "outbox.sqlite")
    with pytest.raises(RuntimeError, match="already consumed"):
        asyncio.run(
            owner.claim_worker_session(
                UUID(bundle.camera_id),
                UUID(expected.generationBinding.workerGenerationId),
                uuid4(),
            )
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows memory-only runtime child binding")
@pytest.mark.parametrize(
    "case",
    [
        "ordinary-startup-with-env",
        "missing-expected",
        "wrong-camera",
        "expired-env",
        "wrong-config-version",
        "wrong-bundle-hash",
    ],
)
def test_windows_run_manifest_rejects_mismatch_or_override_before_model(
    tmp_path, monkeypatch, case
):
    from smartsite_ai.tools import run_camera_runtime as runtime

    value, bundle, cred_path, original, expected, env, manifest_path, profile_path = (
        _setup_runtime_files(tmp_path)
    )
    if case == "expired-env":
        data = json.loads(env[GENERATION_ENV])
        data["expiresAt"] = "2020-01-01T00:00:00.000Z"
        env[GENERATION_ENV] = json.dumps(data)

    for k, v in env.items():
        monkeypatch.setenv(k, v)

    if case == "wrong-config-version":
        expected = expected.model_copy(
            update={
                "configuration": expected.configuration.model_copy(
                    update={"configurationVersion": bundle.configuration_version + 1}
                )
            }
        )
    elif case == "wrong-bundle-hash":
        expected = expected.model_copy(
            update={
                "configuration": expected.configuration.model_copy(
                    update={"bundleSha256": "f" * 64}
                )
            }
        )

    if case == "missing-expected":
        monkeypatch.delenv("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", raising=False)
    else:
        monkeypatch.setenv(
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json()
        )

    requests: list[httpx.Request] = []
    monkeypatch.setattr(
        BackendConfigurationClient,
        "from_settings",
        lambda *args, **kwargs: _bundle_client(bundle, requests),
    )

    model_calls: list[str] = []

    def fake_model(*args, **kwargs):
        model_calls.append("model")
        raise RuntimeError("Model must not load on rejected startup")

    monkeypatch.setattr(runtime, "load_yolo11_detector", fake_model)

    kwargs = (
        {}
        if case == "ordinary-startup-with-env"
        else {
            "source_control_profile": profile_path,
            "expected_camera_id": uuid4() if case == "wrong-camera" else UUID(bundle.camera_id),
        }
    )

    with pytest.raises((CameraRuntimeError, ValueError)):
        asyncio.run(runtime.run_manifest(manifest_path, **kwargs))

    assert model_calls == []
    assert cred_path.read_bytes() == original
    if case in {"ordinary-startup-with-env", "missing-expected", "wrong-camera", "expired-env"}:
        assert requests == []
    else:
        assert len(requests) == 1
        assert GENERATION_ENV not in os.environ
