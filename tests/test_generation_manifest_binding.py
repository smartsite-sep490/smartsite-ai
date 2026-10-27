from pathlib import Path
from uuid import UUID

import pytest
from test_multi_camera_runtime import CAMERA_A, CAMERA_B, _document, _entry

from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
from smartsite_ai.runtime.manifest import CameraRuntimeError, parse_manifest


def manifest(tmp_path):
    entry = _entry(
        "a",
        CAMERA_A,
        "CAM-A",
        source={"kind": "path", "path": "0"},
        outbox=str(tmp_path / "outbox.sqlite3"),
    )
    return parse_manifest(_document(tmp_path, [entry]))


def test_binding_changes_only_generation_file_and_preserves_original(tmp_path):
    value = manifest(tmp_path)
    bound = bind_generation_manifest(
        value, tmp_path / "fresh.json", UUID(CAMERA_A), tmp_path / "outbox.sqlite3"
    )
    assert value.cameras[0].generation_credential_file is None
    actual = bound.model_dump(by_alias=True)
    expected = value.model_dump(by_alias=True)
    expected["cameras"][0]["generationCredentialFile"] = str(tmp_path / "fresh.json")
    assert actual == expected
    assert bind_generation_manifest(value, None, None, None) is value


@pytest.mark.parametrize("mode", ["partial", "camera", "outbox", "relative", "multiple"])
def test_binding_refuses_scope_or_incomplete_override_before_resources(tmp_path, mode):
    value = manifest(tmp_path)
    credential = tmp_path / "fresh.json"
    camera = UUID(CAMERA_A)
    outbox = tmp_path / "outbox.sqlite3"
    if mode == "partial":
        camera = None
    elif mode == "camera":
        camera = UUID(CAMERA_B)
    elif mode == "outbox":
        outbox = tmp_path / "other.sqlite3"
    elif mode == "relative":
        credential = Path("relative.json")
    else:
        entry = _entry(
            "b",
            CAMERA_B,
            "CAM-B",
            source={"kind": "path", "path": "1"},
            outbox=str(tmp_path / "other.sqlite3"),
        )
        value = parse_manifest(
            {
                **value.model_dump(by_alias=True, mode="json"),
                "cameras": [value.model_dump(by_alias=True, mode="json")["cameras"][0], entry],
            }
        )
    with pytest.raises(CameraRuntimeError):
        bind_generation_manifest(value, credential, camera, outbox)


def test_cli_forwards_explicit_binding_without_changing_default_command(tmp_path, monkeypatch):
    from smartsite_ai.tools import run_camera_runtime as cli

    calls = []

    async def run(path, **kwargs):
        calls.append((path, kwargs))
        return 0

    monkeypatch.setattr(cli, "run_manifest", run)
    source = tmp_path / "manifest.json"
    credential = tmp_path / "fresh.json"
    outbox = tmp_path / "outbox.sqlite3"
    assert cli.run(["--manifest", str(source)]) == 0
    assert calls[-1] == (source, {})
    assert (
        cli.run(
            [
                "--manifest",
                str(source),
                "--generation-credential-file",
                str(credential),
                "--expected-camera-id",
                CAMERA_A,
                "--expected-outbox",
                str(outbox),
            ]
        )
        == 0
    )
    assert calls[-1] == (
        source,
        {
            "generation_credential_file": credential,
            "expected_camera_id": UUID(CAMERA_A),
            "expected_outbox": outbox,
        },
    )


def test_actual_run_refuses_wrong_camera_before_preflight_or_model(tmp_path, monkeypatch):
    import asyncio
    import json

    from smartsite_ai.config import Settings
    from smartsite_ai.tools import run_camera_runtime as cli

    value = manifest(tmp_path)
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(value.model_dump(by_alias=True, mode="json")), encoding="utf-8")
    calls = []

    async def never(*args, **kwargs):
        calls.append(True)
        raise AssertionError("resource preflight must not run")

    monkeypatch.setattr(cli, "preflight_manifest", never)
    with pytest.raises(CameraRuntimeError, match="mismatch"):
        asyncio.run(
            cli.run_manifest(
                source,
                settings=Settings(_env_file=None),
                generation_credential_file=tmp_path / "fresh.json",
                expected_camera_id=UUID(CAMERA_B),
                expected_outbox=tmp_path / "outbox.sqlite3",
            )
        )
    assert calls == []
