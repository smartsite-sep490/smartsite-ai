import asyncio
import json
import os
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from test_tracked_runtime_selection import document

from smartsite_ai.integrations.worker_generation import load_worker_generation
from smartsite_ai.runtime.manifest import parse_manifest
from smartsite_ai.runtime.source_control_child import bind_source_control_child
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.source_resume_models import ResumeManifest


def inputs(tmp_path):
    value, bundle = document(tmp_path)
    credential_path = tmp_path / "generation.json"
    original = credential_path.read_bytes()
    old = json.loads(original)
    generation = {
        key: old[key] for key in ("workerGenerationId", "streamSessionId", "secret", "expiresAt")
    }
    generation.update(workerGenerationId=str(uuid4()), streamSessionId=str(uuid4()))
    expected = ResumeManifest(
        generationBinding={
            "cameraId": bundle.camera_id,
            "workerGenerationId": generation["workerGenerationId"],
            "streamSessionId": generation["streamSessionId"],
        },
        configuration={
            "configurationVersion": bundle.configuration_version,
            "viewRevision": bundle.view_revision,
            "bundleSha256": bundle.bundle_sha256,
        },
    )
    env = {
        "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE": "0",
        "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION": json.dumps(generation),
    }
    return value, bundle, credential_path, original, expected, env


@pytest.mark.skipif(os.name == "nt", reason="POSIX private atomic publication only")
def test_explicit_child_profile_binds_saved_source_and_atomically_updates_fixed_credential(
    tmp_path,
):
    value, bundle, path, original, expected, env = inputs(tmp_path)
    manifest = parse_manifest(value)
    bound = bind_source_control_child(
        manifest,
        policy=SourcePolicy({"usb0": 0}, {}, ()),
        scopes={(str(manifest.cameras[0].tracked_v17.site_id), bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )
    assert path.read_bytes() != original
    loaded = load_worker_generation(path, UUID(bundle.camera_id), now=datetime.now(UTC))
    assert loaded.worker_generation_id == expected.generationBinding.workerGenerationId
    assert loaded.stream_session_id == expected.generationBinding.streamSessionId
    assert bound.cameras[0].source.kind == "env"
    assert bound.cameras[0].source.name == "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"
    assert bound.cameras[0].generation_credential_file == str(path)
    assert bound.cameras[0].outbox == manifest.cameras[0].outbox
    assert value["cameras"][0]["source"] == {"kind": "path", "path": "0"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "invalid",
    [
        "camera",
        "binding",
        "scope",
        "source",
        "video",
        "secret",
        "partial",
        "replay",
        "extra-credential-field",
        "expired",
        "untracked",
    ],
)
def test_child_profile_rejects_before_credential_write(tmp_path, invalid, monkeypatch):
    import smartsite_ai.runtime.source_control_child as child

    def never_publish(*args):
        pytest.fail("Rejected input must not reach credential publication")

    monkeypatch.setattr(child, "_persist_credential", never_publish)
    value, bundle, path, original, expected, env = inputs(tmp_path)
    camera = UUID(bundle.camera_id)
    scopes = {(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)}
    if invalid == "camera":
        camera = uuid4()
    if invalid == "binding":
        expected = expected.model_copy(
            update={
                "generationBinding": expected.generationBinding.model_copy(
                    update={"workerGenerationId": str(uuid4())}
                )
            }
        )
    if invalid == "scope":
        scopes = set()
    if invalid == "source":
        env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = "rtsp://169.254.169.254/live"
    if invalid == "video":
        env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = str(tmp_path / "movie.mp4")
    if invalid == "secret":
        data = json.loads(env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"])
        data["secret"] = "x"
        env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"] = json.dumps(data)
    if invalid == "partial":
        env.pop("SMARTSITE_AI_SOURCE_RUNTIME_SOURCE")
    if invalid == "replay":
        value["cameras"][0]["live"] = False
    if invalid == "untracked":
        value["cameras"][0].pop("trackedV17")
    if invalid in {"extra-credential-field", "expired"}:
        data = json.loads(env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"])
        if invalid == "extra-credential-field":
            data["credentialPath"] = str(tmp_path / "arbitrary.json")
        else:
            data["expiresAt"] = "2020-01-01T00:00:00.000Z"
        env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"] = json.dumps(data)
    with pytest.raises((ValueError, RuntimeError)):
        bind_source_control_child(
            parse_manifest(value),
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            scopes=scopes,
            camera_id=camera,
            environ=env,
            expected=expected,
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["disabled", "missing-manifest", "wrong-camera", "exact"])
def test_canonical_entry_profile_validates_binding_before_model(tmp_path, monkeypatch, mode):
    import httpx

    from smartsite_ai.integrations.backend_configuration_client import BackendConfigurationClient
    from smartsite_ai.tools import run_camera_runtime as runtime

    value, bundle, path, original, expected, env = inputs(tmp_path)
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
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    if mode == "missing-manifest":
        monkeypatch.delenv("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", raising=False)
    else:
        monkeypatch.setenv(
            "SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", expected.model_dump_json()
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
        raise RuntimeError("Synthetic model boundary")

    monkeypatch.setattr(runtime, "load_yolo11_detector", model)
    kwargs = (
        {}
        if mode == "disabled"
        else {
            "source_control_profile": profile_path,
            "expected_camera_id": uuid4() if mode == "wrong-camera" else UUID(bundle.camera_id),
        }
    )
    with pytest.raises(RuntimeError):
        asyncio.run(runtime.run_manifest(manifest_path, **kwargs))
    starts = mode == "exact" and os.name != "nt"
    assert calls == (["model"] if starts else [])
    if starts:
        assert (
            load_worker_generation(
                path, UUID(bundle.camera_id), now=datetime.now(UTC)
            ).worker_generation_id
            == expected.generationBinding.workerGenerationId
        )
    else:
        assert path.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="POSIX private atomic publication only")
def test_failed_atomic_credential_replace_preserves_old_file(tmp_path, monkeypatch):
    import smartsite_ai.runtime.source_control_child as child

    value, bundle, path, original, expected, env = inputs(tmp_path)

    def fail(*args):
        raise OSError("Synthetic publication failure")

    monkeypatch.setattr(child.os, "replace", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        bind_source_control_child(
            parse_manifest(value),
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
            camera_id=UUID(bundle.camera_id),
            environ=env,
            expected=expected,
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_configured_rtsp_source_is_accepted_without_credentials_in_manifest(tmp_path, monkeypatch):
    import smartsite_ai.runtime.source_control_child as child

    # Isolate source/binding validation; this is not platform publication evidence.
    published = []
    monkeypatch.setattr(child, "_persist_credential", lambda *args: published.append(args))
    value, bundle, path, original, expected, env = inputs(tmp_path)
    env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = "rtsp://operator:private-password@192.168.1.50/live"
    bound = bind_source_control_child(
        parse_manifest(value),
        policy=SourcePolicy({}, {}, ("192.168.1.0/24",)),
        scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )
    assert "private-password" not in bound.model_dump_json()
    assert "private-password" not in path.read_text()
    assert published[0][0] == path
    assert published[0][1].worker_generation_id == expected.generationBinding.workerGenerationId


def test_cli_forwards_only_explicit_source_control_profile(tmp_path, monkeypatch):
    from smartsite_ai.tools import run_camera_runtime as runtime

    calls = []

    async def run_manifest(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(runtime, "run_manifest", run_manifest)
    profile = tmp_path / "host.json"
    manifest = tmp_path / "runtime.json"
    camera = str(uuid4())
    assert (
        runtime.run(
            [
                "--manifest",
                str(manifest),
                "--source-control-profile",
                str(profile),
                "--expected-camera-id",
                camera,
            ]
        )
        == 0
    )
    assert calls[0][1]["source_control_profile"] == profile
    assert calls[0][1]["expected_camera_id"] == UUID(camera)


@pytest.mark.skipif(os.name != "nt", reason="Windows fail-closed publication gate")
def test_windows_profile_cannot_publish_plaintext_credential(tmp_path, monkeypatch):
    import smartsite_ai.runtime.source_control_child as child

    value, bundle, path, original, expected, env = inputs(tmp_path)

    def never_open(*args):
        pytest.fail("Windows must fail before creating a credential temp file")

    monkeypatch.setattr(child.os, "open", never_open)
    with pytest.raises(RuntimeError, match="unavailable"):
        bind_source_control_child(
            parse_manifest(value),
            policy=SourcePolicy({"usb0": 0}, {}, ()),
            scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
            camera_id=UUID(bundle.camera_id),
            environ=env,
            expected=expected,
        )
    assert path.read_bytes() == original
