import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from smartsite_ai.inference.artifacts import ArtifactValidationError
from smartsite_ai.inference.loading import (
    CANONICAL_PPE_CLASS_MAP,
    ProviderModelMetadata,
    load_artifact_spec,
    load_detector_artifact,
    load_yolo11_detector,
)
from smartsite_ai.inference.yolo import RawYoloDetection


def artifact_document(artifact_path: Path, **overrides: object) -> dict[str, object]:
    sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "artifactId": "smartsite-yolo11s-ppe",
        "version": "2026.09.24",
        "modelFamily": "yolo11s",
        "artifactPath": str(artifact_path),
        "sha256": sha256,
        "sourceUrl": "https://models.example.test/smartsite-yolo11s-ppe.pt",
        "license": "AGPL-3.0",
        "classMap": {str(key): value for key, value in CANONICAL_PPE_CLASS_MAP},
        "confidenceThreshold": 0.25,
        "iouThreshold": 0.45,
        "imageSize": [640, 640],
        "device": "cuda:0",
        **overrides,
    }


def write_spec(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    artifact_path = (tmp_path / "model.pt").resolve()
    artifact_path.write_bytes(b"verified-yolo11s")
    spec_path = tmp_path / "artifact.json"
    spec_path.write_text(
        json.dumps(artifact_document(artifact_path, **overrides)), encoding="utf-8"
    )
    return spec_path, artifact_path


class FakeRunner:
    def __init__(self, metadata: object) -> None:
        self._metadata = metadata
        self.loaded_with: object | None = None
        self.closed = False

    @property
    def provider_metadata(self) -> object:
        return self._metadata

    def load(self, artifact: object) -> None:
        self.loaded_with = artifact

    def predict(self, _frame: object) -> tuple[RawYoloDetection, ...]:
        return ()

    def close(self) -> None:
        self.closed = True


def valid_metadata(**overrides: object) -> dict[str, object]:
    return {
        "providerName": "ultralytics",
        "providerVersion": "8.4.155",
        "architecture": "yolo11",
        "variant": "s",
        "task": "detect",
        "classMap": {str(key): value for key, value in CANONICAL_PPE_CLASS_MAP},
        **overrides,
    }


def test_import_has_no_provider_or_hardware_side_effects() -> None:
    script = (
        "import sys; import smartsite_ai.inference.loading; "
        "assert 'ultralytics' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'cv2' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_load_artifact_spec_accepts_only_the_camel_case_contract(tmp_path: Path) -> None:
    spec_path, artifact_path = write_spec(tmp_path)

    spec = load_artifact_spec(spec_path)

    assert spec.artifact_id == "smartsite-yolo11s-ppe"
    assert spec.artifact_path == artifact_path
    assert spec.model_family == "yolo11s"
    assert spec.class_map == CANONICAL_PPE_CLASS_MAP
    assert spec.image_size == (640, 640)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"artifact_id": "snake-case"}, "Extra inputs are not permitted"),
        ({"artifactId": None}, "artifactId"),
        ({"extra": True}, "Extra inputs are not permitted"),
        ({"modelFamily": "yolov8s"}, "modelFamily"),
        ({"modelFamily": "YOLO11s"}, "modelFamily"),
        (
            {"classMap": {"0": "Person", "1": "Hardhat"}},
            "exactly the five canonical PPE classes",
        ),
        (
            {
                "classMap": {
                    "0": "Person",
                    "1": "Hardhat",
                    "2": "NO-Hardhat",
                    "3": "Safety Vest",
                    "5": "NO-Safety Vest",
                }
            },
            "exactly the five canonical PPE classes",
        ),
        ({"confidenceThreshold": "0.25"}, "confidenceThreshold"),
        ({"imageSize": [640, 0]}, "imageSize"),
        ({"artifactPath": "relative/model.pt"}, "artifactPath"),
        (
            {"sourceUrl": "https://models.example.test/model.pt?token=secret"},
            "sourceUrl",
        ),
        ({"apiKey": "secret"}, "secret-bearing field"),
        ({"accessToken": "secret"}, "secret-bearing field"),
        (
            {
                "classMap": {
                    "0": "Person",
                    "1": "Hardhat",
                    "2": "NO-Hardhat",
                    "3": "Safety Vest",
                    "4": "NO-Safety Vest",
                    "password": "secret",
                }
            },
            "secret-bearing field",
        ),
    ],
)
def test_load_artifact_spec_rejects_invalid_or_secret_bearing_json(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    spec_path, _ = write_spec(tmp_path, **overrides)

    with pytest.raises(ArtifactValidationError, match=match):
        load_artifact_spec(spec_path)


@pytest.mark.parametrize("content", ["[]", "null", "{", "\ufeff{}"])
def test_load_artifact_spec_rejects_non_object_or_malformed_json(
    tmp_path: Path, content: str
) -> None:
    spec_path = tmp_path / "artifact.json"
    spec_path.write_text(content, encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        load_artifact_spec(spec_path)


def test_load_artifact_spec_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    spec_path, artifact_path = write_spec(tmp_path)
    document = json.dumps(artifact_document(artifact_path))
    duplicated = document.replace(
        '"version": "2026.09.24",',
        '"version": "2026.09.24", "version": "tampered",',
    )
    spec_path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="valid JSON"):
        load_artifact_spec(spec_path)


def test_shared_loader_verifies_loads_and_returns_an_immutable_class_map(
    tmp_path: Path,
) -> None:
    spec_path, _ = write_spec(tmp_path)
    runner = FakeRunner(valid_metadata())

    detector, returned_runner, artifact, class_map = load_yolo11_detector(
        spec_path, runner_factory=lambda: runner
    )

    assert returned_runner is runner
    assert runner.loaded_with is artifact
    assert artifact.actual_sha256 == artifact.sha256
    assert detector is not None
    assert dict(class_map) == dict(CANONICAL_PPE_CLASS_MAP)
    with pytest.raises(TypeError):
        class_map[0] = "tampered"  # type: ignore[index]


def test_generic_loader_preserves_smoke_models_without_claiming_yolo11_metadata(
    tmp_path: Path,
) -> None:
    artifact_path = (tmp_path / "reference.pt").resolve()
    artifact_path.write_bytes(b"reference-only")
    from smartsite_ai.inference.artifacts import ModelArtifactSpec

    spec = ModelArtifactSpec(
        artifact_id="yolov8-reference",
        version="smoke-only",
        model_family="yolov8n",
        artifact_path=artifact_path,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        source_url="https://models.example.test/reference.pt",
        license="AGPL-3.0",
        class_map=((0, "Person"),),
        confidence_threshold=0.25,
        iou_threshold=0.45,
        image_size=(640, 640),
        device="cpu",
    )
    runner = FakeRunner(metadata={})

    detector, returned_runner, artifact, class_map = load_detector_artifact(
        spec, runner_factory=lambda: runner
    )

    assert detector is not None
    assert returned_runner is runner
    assert runner.loaded_with is artifact
    assert dict(class_map) == {0: "Person"}


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({}, "provider metadata"),
        (valid_metadata(architecture="yolov8"), "YOLO11"),
        (valid_metadata(task="segment"), "detection task"),
        (valid_metadata(variant="n"), "model family"),
        (
            valid_metadata(
                classMap={
                    "0": "Person",
                    "1": "Hardhat",
                    "2": "NO-Hardhat",
                    "3": "Safety Vest",
                    "4": "NO-Safety Vest altered",
                }
            ),
            "class map",
        ),
        (valid_metadata(providerVersion=""), "providerVersion"),
        (valid_metadata(extra="unexpected"), "Extra inputs are not permitted"),
        (
            valid_metadata(
                classMap={
                    **{str(key): value for key, value in CANONICAL_PPE_CLASS_MAP},
                    "04": "NO-Safety Vest",
                }
            ),
            "class map",
        ),
    ],
)
def test_shared_loader_rejects_absent_or_contradictory_provider_metadata_and_closes(
    tmp_path: Path, metadata: dict[str, object], match: str
) -> None:
    spec_path, _ = write_spec(tmp_path)
    runner = FakeRunner(metadata)

    with pytest.raises(ArtifactValidationError, match=match):
        load_yolo11_detector(spec_path, runner_factory=lambda: runner)

    assert runner.closed is True


def test_provider_metadata_is_strict_frozen_and_camel_case() -> None:
    metadata = ProviderModelMetadata.model_validate(valid_metadata())

    assert metadata.model_family == "yolo11s"
    dumped = metadata.model_dump(by_alias=True)
    assert dumped["providerName"] == "ultralytics"
    assert dumped["classMap"] == {
        str(class_id): class_name for class_id, class_name in CANONICAL_PPE_CLASS_MAP
    }
    with pytest.raises(Exception, match="frozen"):
        metadata.task = "segment"  # type: ignore[misc]


def test_shared_loader_closes_runner_when_load_fails(tmp_path: Path) -> None:
    spec_path, _ = write_spec(tmp_path)

    class FailingRunner(FakeRunner):
        def load(self, artifact: Any) -> None:
            del artifact
            raise RuntimeError("provider failed")

    runner = FailingRunner(valid_metadata())

    with pytest.raises(RuntimeError, match="provider failed"):
        load_yolo11_detector(spec_path, runner_factory=lambda: runner)

    assert runner.closed is True
