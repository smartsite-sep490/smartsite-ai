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


def test_realtime_expanded_config_requires_consumer_and_exact_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from pydantic import ValidationError

    from smartsite_ai import realtime
    from smartsite_ai.config import Settings

    with pytest.raises(ValidationError):
        Settings(realtime_experimental_model_profile="native-ppe-10")
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({"0": "Person", "4": "NO-Safety Vest"}), encoding="utf-8")
    settings = Settings(
        realtime_experimental_model_profile="native-ppe-10",
        realtime_observation_schema_version="1.1.0",
        realtime_class_map_path=str(classes),
        realtime_model_path=str(tmp_path / "missing.pt"),
        realtime_region_configuration_path=str(tmp_path / "missing.json"),
    )

    def no_gpu(*args):
        raise AssertionError("wrong taxonomy must fail before GPU initialization")

    monkeypatch.setattr(realtime, "_configured_realtime_device", no_gpu)
    with pytest.raises(ValueError, match="native-ppe-10"):
        realtime._load_realtime_stack(settings)


def test_expanded_preview_keeps_unobserved_items_unknown_not_compliant() -> None:
    from unittest.mock import MagicMock

    from smartsite_ai import realtime
    from smartsite_ai.inference.ppe_profiles import NATIVE_PPE_10_PROFILE

    event = MagicMock()
    event.to_wire_dict.return_value = {
        "cameraExternalId": "CAM-A",
        "observations": [
            {"type": "PERSON", "trackId": 1, "confidence": 0.9},
            {"type": "PPE", "trackId": 1, "ppeItem": "HARD_HAT", "status": "PRESENT"},
            {"type": "PPE", "trackId": 1, "ppeItem": "SAFETY_VEST", "status": "PRESENT"},
        ],
    }
    payload = realtime._ui_frame(event, 640, 480, ppe_items=NATIVE_PPE_10_PROFILE.present_items)
    result = payload["detections"][0]
    assert result["ppeStatus"]["GLOVES"] == "UNKNOWN"
    assert result["ppeStatus"]["BOOTS"] == "UNKNOWN"
    assert result["ppeStatus"]["GOGGLES"] == "UNKNOWN"
    assert result["alertState"] == "UNKNOWN"


@pytest.mark.parametrize("expanded", [False, True])
def test_realtime_uses_versioned_artifact_preprocessing_without_legacy_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expanded: bool
) -> None:
    from unittest.mock import MagicMock

    from test_mf05_mf06_pipeline import PPE_REGION_ID, configuration
    from test_ppe_model_profiles import NATIVE_MAP, document

    from smartsite_ai import realtime
    from smartsite_ai.config import Settings
    from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP
    from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner

    region_path = tmp_path / "regions.json"
    region_path.write_text(configuration().model_dump_json(by_alias=True), encoding="utf-8")
    settings = Settings(
        realtime_artifact_spec_path=str(
            document(
                tmp_path,
                NATIVE_MAP if expanded else dict((str(k), v) for k, v in CANONICAL_PPE_CLASS_MAP),
            )
        ),
        realtime_experimental_model_profile="native-ppe-10" if expanded else None,
        realtime_observation_schema_version="1.1.0" if expanded else "1.0.0",
        realtime_device="cpu",
        realtime_region_configuration_path=str(region_path),
        realtime_camera_external_id="camera-01",
        realtime_ppe_region_id=PPE_REGION_ID,
    )
    runner = MagicMock(spec=UltralyticsYoloRunner)

    def load(spec, **kwargs):
        assert spec.image_size == (832, 832)
        assert spec.iou_threshold == 0.7
        assert spec.confidence_threshold == 0.25
        assert dict(spec.class_map)[4] == ("Gloves" if expanded else "NO-Safety Vest")
        assert spec.device == "cpu"
        assert kwargs["require_yolo11_metadata"] is True
        return MagicMock(), runner, MagicMock(), dict(spec.class_map)

    monkeypatch.setattr(realtime, "_configured_realtime_device", lambda device: device)
    monkeypatch.setattr(realtime, "load_detector_artifact", load)
    _, actual_runner, _, _ = realtime._load_realtime_stack(settings)
    assert actual_runner is runner


def test_realtime_artifact_rejects_conflicting_raw_weight_or_taxonomy(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from smartsite_ai.config import Settings

    for key in ["realtime_model_path", "realtime_class_map_path"]:
        with pytest.raises(ValidationError, match="artifact spec"):
            Settings(realtime_artifact_spec_path=tmp_path / "artifact.json", **{key: "other"})


def test_realtime_route_accepts_artifact_configuration_without_raw_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from smartsite_ai import app as app_module
    from smartsite_ai.config import Settings

    async def stream(socket, settings):
        await socket.accept()
        await socket.send_json(
            {"type": "fixture", "artifact": str(settings.realtime_artifact_spec_path)}
        )
        await socket.close()

    monkeypatch.setattr(app_module, "stream_realtime", stream)
    settings = Settings(
        backend_service_token="fixture-secret",
        realtime_artifact_spec_path=tmp_path / "artifact.json",
        realtime_source="fixture.mp4",
    )
    with (
        TestClient(app_module.create_app(settings)) as client,
        client.websocket_connect("/ws/realtime?token=fixture-secret") as socket,
    ):
        assert socket.receive_json()["type"] == "fixture"
