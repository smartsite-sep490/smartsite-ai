import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.ultralytics_runner import (
    UltralyticsYoloRunner,
    _bgr_image,
    _default_model_factory,
)
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    InferenceResultError,
    RawYoloDetection,
    Yolo11Detector,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

ROOT = Path(__file__).resolve().parents[1]
FAKE_MODEL_PATH = (ROOT / "models" / "yolo11s-ppe.pt").resolve()
MODEL_SHA256 = "a" * 64
SESSION = UUID("00000000-0000-4000-8000-000000000001")


def make_artifact(**overrides: object) -> VerifiedModelArtifact:
    return VerifiedModelArtifact.model_validate(
        {
            "artifact_id": "yolo11s-ppe",
            "version": "2026.09.21",
            "model_family": "yolo11s",
            "artifact_path": FAKE_MODEL_PATH,
            "sha256": MODEL_SHA256,
            "source_url": "https://models.example.test/yolo11s-ppe.pt",
            "license": "AGPL-3.0",
            "class_map": ((0, "person"), (1, "hard_hat")),
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "image_size": (640, 640),
            "device": "cuda:0",
            "resolved_path": FAKE_MODEL_PATH,
            "actual_sha256": MODEL_SHA256,
            **overrides,
        }
    )


def make_frame(**overrides: object) -> FrameEnvelope:
    data: dict[str, object] = {
        "stream_id": "stream-01",
        "session_id": SESSION,
        "camera_external_id": "cam-01",
        "captured_at": datetime(2026, 9, 21, 12, tzinfo=UTC),
        "width": 4,
        "height": 2,
        "sequence_number": 4,
        "payload": bytes(range(24)),
        **overrides,
    }
    return FrameEnvelope.model_validate(data)


class FakeVector:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def tolist(self) -> list[object]:
        return self._values


class FakeBoxes:
    def __init__(
        self,
        *,
        xyxy: list[list[float]],
        confidence: list[float],
        class_ids: list[float],
    ) -> None:
        self.xyxy = FakeVector(xyxy)
        self.conf = FakeVector(confidence)
        self.cls = FakeVector(class_ids)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self._results = results
        self.calls: list[tuple[object, dict[str, object]]] = []

    def predict(self, source: object, **kwargs: object) -> list[FakeResult]:
        self.calls.append((source, kwargs))
        return self._results


class FakeProviderNetwork:
    def __init__(self, *, scale: str = "s", yaml_file: str = "yolo11.yaml") -> None:
        self.yaml = {"scale": scale, "yaml_file": yaml_file}


class FakeProviderModel(FakeModel):
    def __init__(
        self,
        results: list[FakeResult],
        *,
        scale: str = "s",
        yaml_file: str = "yolo11.yaml",
        task: str = "detect",
        names: dict[int, str] | None = None,
    ) -> None:
        super().__init__(results)
        self.model = FakeProviderNetwork(scale=scale, yaml_file=yaml_file)
        self.task = task
        self.names = names or {0: "person", 1: "hard_hat"}


class FakeImageFlags:
    writeable = False


class FakeImage:
    def __init__(self, frame: FrameEnvelope) -> None:
        self.shape = (frame.height, frame.width, 3)
        self._buffer = frame.buffer
        self.flags = FakeImageFlags()

    def tobytes(self) -> bytes:
        return self._buffer


def fake_bgr_image(frame: FrameEnvelope) -> FakeImage:
    return FakeImage(frame)


def make_result(*, names: dict[int, str] | None = None) -> FakeResult:
    return FakeResult(
        FakeBoxes(
            xyxy=[[1.5, 2.5, 3.5, 4.5], [5.0, 6.0, 7.0, 8.0]],
            confidence=[0.9, 0.8],
            class_ids=[1.0, 0.0],
        ),
        names or {0: "person", 1: "hard_hat"},
    )


def test_importing_runner_does_not_load_provider_runtime() -> None:
    environment = os.environ.copy()
    python_path = str(ROOT / "src")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path
    code = (
        "import sys; "
        "import smartsite_ai.inference.ultralytics_runner; "
        "loaded = {'ultralytics', 'torch', 'cv2'} & set(sys.modules); "
        "assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_bgr_image_returns_a_read_only_bgr24_view() -> None:
    numpy = pytest.importorskip("numpy")
    frame = make_frame()

    image = _bgr_image(frame)

    assert image.dtype == numpy.uint8
    assert image.shape == (2, 4, 3)
    assert image.tobytes() == bytes(range(24))
    assert image.tolist() == [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
    ]
    assert not image.flags.writeable
    with pytest.raises(ValueError):
        image[0, 0, 0] = 255
    assert frame.buffer == bytes(range(24))


def test_load_is_explicit_and_prediction_uses_verified_path_and_frame_configuration() -> None:
    artifact = make_artifact()
    model = FakeModel([make_result()])
    constructed_paths: list[Path] = []

    def factory(path: Path) -> FakeModel:
        constructed_paths.append(path)
        return model

    runner = UltralyticsYoloRunner(model_factory=factory, image_factory=fake_bgr_image)

    assert constructed_paths == []

    runner.load(artifact)
    rows = runner.predict(make_frame())

    assert constructed_paths == [FAKE_MODEL_PATH]
    assert rows == (
        RawYoloDetection(class_id=1, confidence=0.9, x1=1.5, y1=2.5, x2=3.5, y2=4.5),
        RawYoloDetection(class_id=0, confidence=0.8, x1=5.0, y1=6.0, x2=7.0, y2=8.0),
    )
    source, options = model.calls[0]
    assert source.shape == (2, 4, 3)
    assert source.tobytes() == bytes(range(24))
    assert not source.flags.writeable
    assert options == {
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": (640, 640),
        "device": "cuda:0",
        "verbose": False,
    }


def test_provider_metadata_is_derived_from_loaded_checkpoint_facts() -> None:
    model = FakeProviderModel([make_result()], yaml_file="yolo11s.yaml")
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: model,
        image_factory=fake_bgr_image,
        provider_version_factory=lambda: "8.4.155",
    )

    runner.load(make_artifact())

    assert runner.provider_metadata == {
        "providerName": "ultralytics",
        "providerVersion": "8.4.155",
        "architecture": "yolo11",
        "variant": "s",
        "task": "detect",
        "classMap": {"0": "person", "1": "hard_hat"},
    }


def test_provider_metadata_accepts_canonical_family_yaml_with_separate_scale() -> None:
    model = FakeProviderModel([make_result()], scale="s", yaml_file="yolo11.yaml")
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: model,
        image_factory=fake_bgr_image,
        provider_version_factory=lambda: "8.4.155",
    )

    runner.load(make_artifact())

    assert runner.provider_metadata["architecture"] == "yolo11"
    assert runner.provider_metadata["variant"] == "s"


def test_provider_metadata_rejects_yaml_variant_that_conflicts_with_scale() -> None:
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: FakeProviderModel([], scale="s", yaml_file="yolo11n.yaml"),
        provider_version_factory=lambda: "8.4.155",
    )
    runner.load(make_artifact())

    with pytest.raises(DetectorUnavailableError, match="variant metadata is inconsistent"):
        _ = runner.provider_metadata


def test_provider_metadata_rejects_unproven_or_contradictory_checkpoint_facts() -> None:
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: FakeProviderModel([], scale="s", yaml_file="yolov8.yaml"),
        provider_version_factory=lambda: "8.4.155",
    )
    runner.load(make_artifact())

    with pytest.raises(DetectorUnavailableError, match="YOLO11 architecture"):
        _ = runner.provider_metadata


def test_provider_metadata_before_load_is_rejected() -> None:
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: FakeProviderModel([]),
        provider_version_factory=lambda: "8.4.155",
    )

    with pytest.raises(DetectorUnavailableError, match="not loaded"):
        _ = runner.provider_metadata


def test_predict_rejects_a_provider_class_name_that_disagrees_with_the_artifact() -> None:
    runner = UltralyticsYoloRunner(
        model_factory=lambda _: FakeModel([make_result(names={0: "person", 1: "helmet"})]),
        image_factory=fake_bgr_image,
    )
    runner.load(make_artifact())

    with pytest.raises(InferenceResultError, match="class name"):
        runner.predict(make_frame())


def test_predict_before_load_is_rejected() -> None:
    runner = UltralyticsYoloRunner(model_factory=lambda _: FakeModel([]))

    with pytest.raises(DetectorUnavailableError, match="not loaded"):
        runner.predict(make_frame())


def test_repeated_load_is_rejected() -> None:
    runner = UltralyticsYoloRunner(model_factory=lambda _: FakeModel([]))
    runner.load(make_artifact())

    with pytest.raises(DetectorUnavailableError, match="already loaded"):
        runner.load(make_artifact())


def test_provider_prediction_exception_is_translated() -> None:
    class FailingModel:
        def predict(self, source: object, **kwargs: object) -> list[FakeResult]:
            raise RuntimeError("CUDA unavailable")

    runner = UltralyticsYoloRunner(
        model_factory=lambda _: FailingModel(), image_factory=fake_bgr_image
    )
    runner.load(make_artifact())

    with pytest.raises(DetectorUnavailableError, match="provider prediction failed") as exc_info:
        runner.predict(make_frame())

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_close_releases_references_and_allows_a_new_explicit_load() -> None:
    models = [FakeModel([]), FakeModel([])]
    runner = UltralyticsYoloRunner(model_factory=lambda _: models.pop(0))
    artifact = make_artifact()
    runner.load(artifact)

    runner.close()

    with pytest.raises(DetectorUnavailableError, match="not loaded"):
        runner.predict(make_frame())
    runner.load(artifact)


def test_rectangular_image_size_is_passed_as_provider_height_then_width() -> None:
    artifact = make_artifact(image_size=(1280, 736))
    model = FakeModel([make_result()])
    runner = UltralyticsYoloRunner(model_factory=lambda _: model, image_factory=fake_bgr_image)

    runner.load(artifact)
    runner.predict(make_frame())

    assert model.calls[0][1]["imgsz"] == (736, 1280)


def test_composed_detector_preserves_provider_result_errors() -> None:
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes(
                    xyxy=[[1.0, 1.0, 2.0, 2.0]],
                    confidence=[float("nan")],
                    class_ids=[0.0],
                ),
                {0: "person", 1: "hard_hat"},
            )
        ]
    )
    runner = UltralyticsYoloRunner(model_factory=lambda _: model, image_factory=fake_bgr_image)
    runner.load(make_artifact())
    detector = Yolo11Detector(make_artifact(), runner)

    with pytest.raises(InferenceResultError, match="finite"):
        asyncio.run(detector.detect(make_frame()))


def test_removed_verified_file_fails_without_opening_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")
    artifact.unlink()

    def refuse_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider opened the network")

    monkeypatch.setattr("urllib.request.urlopen", refuse_network)

    with pytest.raises(DetectorUnavailableError, match="unavailable"):
        _default_model_factory(artifact)


def test_apostrophe_path_reaches_the_provider_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ultralytics")
    real_dir = tmp_path / "O'Brien"
    real_dir.mkdir()
    real = real_dir / "model.pt"
    real.write_bytes(b"exact-bytes")
    decoy_dir = tmp_path / "OBrien"
    decoy_dir.mkdir()
    decoy = decoy_dir / "model.pt"
    decoy.write_bytes(b"decoy-bytes")
    import ultralytics.nn.tasks as tasks
    import ultralytics.utils.downloads as downloads

    loaded: list[Path] = []

    def record_load(file: object, *_args: object, **_kwargs: object) -> None:
        loaded.append(Path(str(file)))
        raise RuntimeError("stop after the exact path is selected")

    def refuse_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider attempted a download")

    monkeypatch.setattr(tasks, "torch_load", record_load)
    monkeypatch.setattr(downloads, "safe_download", refuse_download)

    with pytest.raises(DetectorUnavailableError, match="construction failed"):
        _default_model_factory(real)

    assert len(loaded) == 1
    assert os.path.normcase(str(loaded[0])) == os.path.normcase(str(real))
    assert decoy.read_bytes() == b"decoy-bytes"


def test_missing_checkpoint_dependency_does_not_autoinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ultralytics")
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"placeholder")
    import ultralytics.nn.tasks as tasks

    def missing_dependency(*_args: object, **_kwargs: object) -> None:
        raise ModuleNotFoundError(
            "No module named 'smartsite_missing_provider_dep'",
            name="smartsite_missing_provider_dep",
        )

    def refuse_install(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider attempted to install a package")

    monkeypatch.setattr(tasks, "torch_load", missing_dependency)
    monkeypatch.setattr(subprocess, "check_output", refuse_install)

    with pytest.raises(DetectorUnavailableError, match="auto-install"):
        _default_model_factory(artifact)
