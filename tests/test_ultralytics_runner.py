import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    InferenceResultError,
    RawYoloDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "a" * 64
SESSION = UUID("00000000-0000-4000-8000-000000000001")


def make_artifact(**overrides: object) -> VerifiedModelArtifact:
    return VerifiedModelArtifact.model_validate(
        {
            "artifact_id": "yolo11s-ppe",
            "version": "2026.09.21",
            "artifact_path": Path("C:/models/yolo11s-ppe.pt"),
            "sha256": MODEL_SHA256,
            "source_url": "https://models.example.test/yolo11s-ppe.pt",
            "license": "AGPL-3.0",
            "class_map": ((0, "person"), (1, "hard_hat")),
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "image_size": (640, 640),
            "device": "cuda:0",
            "resolved_path": Path("C:/models/yolo11s-ppe.pt"),
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

    assert constructed_paths == [Path("C:/models/yolo11s-ppe.pt")]
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
