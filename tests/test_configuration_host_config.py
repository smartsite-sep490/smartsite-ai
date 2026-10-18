"""Private operator input validation; no source/server/model construction."""

import importlib
import importlib.util
import json

import pytest

SOURCE = "rtsp://user:source-secret@example.invalid/live?key=private"
TOKEN = "private-configuration-service-token-12345"


def api():
    name = "smartsite_ai.tools.configuration_host_config"
    assert importlib.util.find_spec(name) is not None, "Explicit host configuration is missing"
    return importlib.import_module(name)


def manifest(tmp_path, **changes):
    value = {
        "schemaVersion": "1.0.0",
        "streamId": "configuration-only",
        "cameraExternalId": "LAPTOP-TEST",
        "sourceKind": "LIVE",
        "sourceEnv": "SMARTSITE_AI_CAPTURE_SOURCE",
        "tokenEnv": "SMARTSITE_AI_CAPTURE_TOKEN",
        "resourceId": "e98f6475-0660-4c71-a466-4c93d1739ee0",
        "lockDirectory": str(tmp_path),
        "port": 0,
        "durationSeconds": 0.1,
    }
    value.update(changes)
    path = tmp_path / "private-config.json"
    path.write_text(json.dumps(value))
    return path


def load(path):
    return api().load_host_configuration(
        path,
        {
            "SMARTSITE_AI_CAPTURE_SOURCE": SOURCE,
            "SMARTSITE_AI_CAPTURE_TOKEN": TOKEN,
        },
    )


def test_named_environment_values_stay_secret_and_kind_is_explicit(tmp_path):
    cfg = load(manifest(tmp_path))
    assert cfg.stream.get_raw_source_url() == SOURCE
    assert cfg.token.get_secret_value() == TOKEN
    assert cfg.source_kind == "LIVE" and cfg.stream.is_live
    assert cfg.lock_directory == tmp_path and cfg.port == 0
    assert cfg.duration_seconds == 0.1
    assert SOURCE not in repr(cfg) and TOKEN not in repr(cfg)


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": True},
        {"schemaVersion": "2.0.0"},
        {"sourceEnv": "PATH"},
        {"tokenEnv": "PASSWORD"},
        {"port": True},
        {"port": 65536},
        {"durationSeconds": 0},
        {"durationSeconds": 3601},
        {"durationSeconds": float("nan")},
        {"resourceId": "invalid"},
        {"cameraExternalId": "bad\nidentity"},
    ],
    ids=[
        "unknown",
        "version",
        "source-env",
        "token-env",
        "bool-port",
        "port",
        "zero-duration",
        "duration",
        "nonfinite",
        "resource",
        "identity",
    ],
)
def test_invalid_manifest_is_refused_without_values(tmp_path, changes):
    module = api()
    with pytest.raises(module.ConfigurationHostError) as error:
        load(manifest(tmp_path, **changes))
    assert error.value.code == "CONFIGURATION_HOST_INVALID_CONFIG"
    assert SOURCE not in str(error.value) and TOKEN not in str(error.value)


def test_duplicate_json_keys_and_oversized_files_are_refused(tmp_path):
    module = api()
    path = manifest(tmp_path)
    path.write_text(path.read_text().replace('"port": 0', '"port": 0, "port": 1'))
    with pytest.raises(module.ConfigurationHostError):
        load(path)
    path.write_text(" " * 16385)
    with pytest.raises(module.ConfigurationHostError):
        load(path)


def test_missing_environment_and_unsafe_token_are_refused(tmp_path):
    module = api()
    path = manifest(tmp_path)
    for env in (
        {},
        {"SMARTSITE_AI_CAPTURE_SOURCE": SOURCE, "SMARTSITE_AI_CAPTURE_TOKEN": "bad token"},
    ):
        with pytest.raises(module.ConfigurationHostError) as error:
            module.load_host_configuration(path, env)
        assert str(error.value) == "CONFIGURATION_HOST_INVALID_CONFIG"


def test_local_video_requires_replay_and_replay_is_paced(tmp_path):
    module = api()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not decoded in config test")
    env = {"SMARTSITE_AI_CAPTURE_SOURCE": str(video), "SMARTSITE_AI_CAPTURE_TOKEN": TOKEN}
    with pytest.raises(module.ConfigurationHostError):
        module.load_host_configuration(manifest(tmp_path), env)
    cfg = module.load_host_configuration(manifest(tmp_path, sourceKind="REPLAY"), env)
    assert not cfg.stream.is_live and cfg.stream.pace_replay
    env["SMARTSITE_AI_CAPTURE_SOURCE"] = "0"
    with pytest.raises(module.ConfigurationHostError):
        module.load_host_configuration(manifest(tmp_path, sourceKind="REPLAY"), env)


def test_non_regular_missing_and_symlink_configuration_refused(tmp_path):
    module = api()
    for path in (tmp_path, tmp_path / "missing.json"):
        with pytest.raises(module.ConfigurationHostError):
            load(path)
    original = manifest(tmp_path)
    link = tmp_path / "link.json"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("Host does not permit creating a test symlink")
    with pytest.raises(module.ConfigurationHostError):
        load(link)
