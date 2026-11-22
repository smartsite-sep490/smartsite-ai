import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartsite_ai.inference.artifacts import (
    ArtifactChecksumMismatchError,
    ArtifactNotFoundError,
    ArtifactValidationError,
)
from smartsite_ai.inference.loading import (
    CANONICAL_COCO_80_CLASS_MAP,
    CANONICAL_PPE_CLASS_MAP,
    load_artifact_spec,
    load_person_artifact_spec,
    load_yolo11_detector,
    load_yolo11_person_detector,
)
from smartsite_ai.inference.yolo import RawYoloDetection


def coco_class_map_json() -> dict[str, str]:
    return {str(class_id): class_name for class_id, class_name in CANONICAL_COCO_80_CLASS_MAP}


def person_artifact_document(artifact_path: Path, **overrides: object) -> dict[str, object]:
    sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "artifactId": "smartsite-yolo11s-coco-person",
        "version": "yolo11s-coco-official",
        "modelFamily": "yolo11s",
        "artifactPath": str(artifact_path),
        "sha256": sha256,
        "sourceUrl": "https://github.com/ultralytics/ultralytics",
        "license": "AGPL-3.0",
        "classMap": coco_class_map_json(),
        "confidenceThreshold": 0.25,
        "iouThreshold": 0.7,
        "imageSize": [832, 832],
        "device": "cuda:0",
        **overrides,
    }


def write_person_spec(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    artifact_path = (tmp_path / "yolo11s-coco.pt").resolve()
    artifact_path.write_bytes(b"verified-yolo11s-coco-80")
    spec_path = tmp_path / "person-artifact.json"
    spec_path.write_text(
        json.dumps(person_artifact_document(artifact_path, **overrides)),
        encoding="utf-8",
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


def valid_coco_metadata(**overrides: object) -> dict[str, object]:
    return {
        "providerName": "ultralytics",
        "providerVersion": "8.4.155",
        "architecture": "yolo11",
        "variant": "s",
        "task": "detect",
        "classMap": coco_class_map_json(),
        **overrides,
    }


def test_person_loader_import_has_no_provider_or_hardware_side_effects() -> None:
    script = (
        "import sys; "
        "from smartsite_ai.inference.loading import load_yolo11_person_detector; "
        "assert callable(load_yolo11_person_detector); "
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


def test_load_yolo11_person_detector_loads_canonical_coco80_and_verifies_provider_metadata(
    tmp_path: Path,
) -> None:
    spec_path, artifact_path = write_person_spec(tmp_path)
    runner = FakeRunner(valid_coco_metadata())

    detector, loaded_runner, artifact, class_map = load_yolo11_person_detector(
        spec_path,
        runner_factory=lambda: runner,
    )

    assert detector is not None
    assert loaded_runner is runner
    assert runner.loaded_with == artifact
    assert not runner.closed
    assert artifact.resolved_path == artifact_path
    assert len(class_map) == 80
    assert class_map[0] == "person"
    assert class_map[79] == "toothbrush"


def test_default_ppe_loader_still_rejects_coco80_spec(tmp_path: Path) -> None:
    spec_path, _ = write_person_spec(tmp_path)

    with pytest.raises(ArtifactValidationError, match="exactly the five canonical PPE classes"):
        load_artifact_spec(spec_path)

    runner = FakeRunner(valid_coco_metadata())
    with pytest.raises(ArtifactValidationError, match="exactly the five canonical PPE classes"):
        load_yolo11_detector(spec_path, runner_factory=lambda: runner)
    assert runner.loaded_with is None


def test_person_loader_rejects_ppe_spec_and_non_canonical_coco_class_maps(tmp_path: Path) -> None:
    ppe_spec_path, _ = write_person_spec(
        tmp_path,
        classMap={str(key): name for key, name in CANONICAL_PPE_CLASS_MAP},
    )
    with pytest.raises(ArtifactValidationError, match="COCO"):
        load_person_artifact_spec(ppe_spec_path)

    missing_class = coco_class_map_json()
    del missing_class["79"]
    missing_path, _ = write_person_spec(tmp_path, classMap=missing_class)
    with pytest.raises(ArtifactValidationError, match="COCO"):
        load_person_artifact_spec(missing_path)

    renamed_person = coco_class_map_json()
    renamed_person["0"] = "Person"
    renamed_path, _ = write_person_spec(tmp_path, classMap=renamed_person)
    with pytest.raises(ArtifactValidationError, match="COCO"):
        load_person_artifact_spec(renamed_path)

    extra_class = coco_class_map_json()
    extra_class["80"] = "hardhat"
    extra_path, _ = write_person_spec(tmp_path, classMap=extra_class)
    with pytest.raises(ArtifactValidationError, match="COCO"):
        load_person_artifact_spec(extra_path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"extra": True}, "Extra inputs are not permitted"),
        ({"confidenceThreshold": True}, "confidenceThreshold"),
        ({"iouThreshold": False}, "iouThreshold"),
        ({"confidenceThreshold": float("nan")}, "invalid"),
        ({"artifactPath": "relative/yolo11s.pt"}, "artifactPath"),
        ({"modelFamily": "yolov8s"}, "modelFamily"),
        ({"apiKey": "secret"}, "secret-bearing field"),
    ],
)
def test_person_spec_rejects_invalid_bool_nonfinite_or_secret_fields(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    artifact_path = (tmp_path / "yolo11s-coco.pt").resolve()
    artifact_path.write_bytes(b"verified-yolo11s-coco-80")
    spec_path = tmp_path / "bad-person-artifact.json"
    doc = person_artifact_document(artifact_path, **overrides)
    spec_path.write_text(json.dumps(doc, allow_nan=True), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=match):
        load_person_artifact_spec(spec_path)


def test_person_loader_rejects_missing_file_or_checksum_mismatch_before_loading_runner(
    tmp_path: Path,
) -> None:
    spec_path, artifact_path = write_person_spec(tmp_path)
    artifact_path.unlink()
    runner = FakeRunner(valid_coco_metadata())
    with pytest.raises(ArtifactNotFoundError):
        load_yolo11_person_detector(spec_path, runner_factory=lambda: runner)
    assert runner.loaded_with is None

    bad_sha_spec, _ = write_person_spec(tmp_path, sha256="0" * 64)
    runner2 = FakeRunner(valid_coco_metadata())
    with pytest.raises(ArtifactChecksumMismatchError):
        load_yolo11_person_detector(bad_sha_spec, runner_factory=lambda: runner2)
    assert runner2.loaded_with is None


@pytest.mark.parametrize(
    ("metadata_overrides", "match"),
    [
        ({"architecture": "yolov8"}, "YOLO11 architecture"),
        ({"task": "segment"}, "detection task"),
        ({"variant": "m"}, "model family contradicts"),
        ({"classMap": {**coco_class_map_json(), "56": "sofa"}}, "class map contradicts"),
        (
            {"classMap": {str(k): v for k, v in CANONICAL_PPE_CLASS_MAP}},
            "class map contradicts",
        ),
    ],
)
def test_person_loader_rejects_contradictory_provider_metadata_and_closes_runner(
    tmp_path: Path, metadata_overrides: dict[str, object], match: str
) -> None:
    spec_path, _ = write_person_spec(tmp_path)
    runner = FakeRunner(valid_coco_metadata(**metadata_overrides))

    with pytest.raises(ArtifactValidationError, match=match):
        load_yolo11_person_detector(spec_path, runner_factory=lambda: runner)

    assert runner.closed
