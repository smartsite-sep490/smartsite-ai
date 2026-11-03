"""Expanded evidence is opt-in at both artifact and consumer boundaries."""

from pathlib import Path

import pytest
from test_multi_camera_runtime import CAMERA_A, _document, _entry, _video

from smartsite_ai.runtime.manifest import CameraRuntimeError, parse_manifest


def manifest(tmp_path: Path) -> dict[str, object]:
    return _document(
        tmp_path,
        [
            _entry(
                "test",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": _video(tmp_path, "a.mp4")},
                outbox=str(tmp_path / "events.sqlite3"),
            )
        ],
    )


def test_legacy_manifest_keeps_legacy_profile_and_consumer(tmp_path: Path) -> None:
    parsed = parse_manifest(manifest(tmp_path))
    assert parsed.experimental_model_profile is None
    assert parsed.observation_schema_version == "1.0.0"


def test_expanded_manifest_requires_explicit_consumer_version(tmp_path: Path) -> None:
    document = manifest(tmp_path)
    document["experimentalModelProfile"] = "native-ppe-10"
    with pytest.raises(CameraRuntimeError):
        parse_manifest(document)
    document["observationSchemaVersion"] = "1.1.0"
    parsed = parse_manifest(document)
    assert parsed.experimental_model_profile == "native-ppe-10"
    assert parsed.observation_schema_version == "1.1.0"


@pytest.mark.parametrize(
    "key,value", [("experimentalModelProfile", "auto"), ("observationSchemaVersion", "2.0.0")]
)
def test_runtime_does_not_guess_profile_or_consumer(tmp_path: Path, key: str, value: str) -> None:
    document = manifest(tmp_path)
    document[key] = value
    with pytest.raises(CameraRuntimeError):
        parse_manifest(document)


def test_expanded_loader_forwards_profile_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from smartsite_ai.inference import loading

    calls = []
    spec = object()
    sentinel = object()

    def read(path: Path, *, experimental_profile=None):
        calls.append((path, experimental_profile))
        return spec

    def load(actual, **kwargs):
        assert actual is spec
        assert kwargs["require_yolo11_metadata"] is True
        return sentinel

    monkeypatch.setattr(loading, "load_artifact_spec", read)
    monkeypatch.setattr(loading, "load_detector_artifact", load)
    path = tmp_path / "candidate.json"
    assert (
        loading.load_yolo11_detector(
            path, runner_factory=lambda: None, experimental_profile="native-ppe-10"
        )
        is sentinel
    )
    assert calls == [(path, "native-ppe-10")]
