import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from smartsite_ai.evaluation.models import EvaluationFrame
from smartsite_ai.evaluation.overlay import (
    OverlayBoxInstruction,
    OverlayPlan,
    OverlayRelationInstruction,
)
from smartsite_ai.evaluation.provider_validation import (
    PINNED_ULTRALYTICS_VERSION,
    ProviderValidationArguments,
)
from smartsite_ai.evaluation.runtime_adapters import (
    EvaluationRuntimeError,
    OpenCvEvaluationFrameReader,
    OpenCvEvaluationMedia,
    OpenCvEvidenceExporter,
    OpenCvOverlayRenderer,
    UltralyticsValidationProvider,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

np = pytest.importorskip("numpy")


def _image_frame(**overrides: object) -> EvaluationFrame:
    values: dict[str, object] = {
        "frameId": "frame-1",
        "mediaPath": "media/frame.jpg",
        "sha256": "a" * 64,
        "width": 2,
        "height": 2,
        "annotations": [],
        **overrides,
    }
    return EvaluationFrame.model_validate(values)


def _video_frame(**overrides: object) -> EvaluationFrame:
    return _image_frame(
        mediaPath="media/clip.mp4",
        frameIndex=7,
        videoTimeSeconds=0.7,
        **overrides,
    )


def _read(reader: OpenCvEvaluationFrameReader, root: Path, frame: EvaluationFrame) -> FrameEnvelope:
    return reader.read(
        root,
        frame,
        stream_id="evaluation:test",
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
        camera_external_id="evaluation-camera",
        captured_at=datetime(2026, 9, 24, tzinfo=UTC),
        sequence_number=7,
    )


class FakeCapture:
    def __init__(self, image: np.ndarray | None, *, opened: bool = True, position: float = 8.0):
        self.image = image
        self.opened = opened
        self.position = position
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV compatibility
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.image is not None, self.image

    def get(self, prop: int) -> float:
        del prop
        return self.position

    def release(self) -> None:
        self.released = True


class FakeCv2:
    IMREAD_COLOR = 1
    CAP_PROP_POS_FRAMES = 2
    FONT_HERSHEY_SIMPLEX = 3
    LINE_AA = 4

    def __init__(self, image: np.ndarray | None = None, capture: FakeCapture | None = None):
        self.image = image
        self.capture = capture
        self.imread_calls: list[tuple[str, int]] = []
        self.capture_paths: list[str] = []
        self.rectangle_calls: list[tuple[object, ...]] = []
        self.line_calls: list[tuple[object, ...]] = []
        self.text_calls: list[tuple[object, ...]] = []
        self.imwrite_calls: list[tuple[str, np.ndarray]] = []

    def imread(self, path: str, mode: int) -> np.ndarray | None:
        self.imread_calls.append((path, mode))
        return self.image

    def VideoCapture(self, path: str) -> FakeCapture:  # noqa: N802 - OpenCV compatibility
        self.capture_paths.append(path)
        assert self.capture is not None
        return self.capture

    def rectangle(self, *args: object) -> None:
        self.rectangle_calls.append(args)

    def line(self, *args: object) -> None:
        self.line_calls.append(args)

    def putText(self, *args: object) -> None:  # noqa: N802 - OpenCV compatibility
        self.text_calls.append(args)

    def imwrite(self, path: str, image: np.ndarray) -> bool:
        self.imwrite_calls.append((path, image.copy()))
        return True


def test_runtime_module_import_is_side_effect_free() -> None:
    script = (
        "import sys; import smartsite_ai.evaluation.runtime_adapters; "
        "assert 'cv2' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'ultralytics' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_reader_loads_image_as_owned_bgr24_envelope(tmp_path: Path) -> None:
    media = tmp_path / "media" / "frame.jpg"
    media.parent.mkdir()
    media.write_bytes(b"image")
    image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    fake_cv2 = FakeCv2(image=image)

    envelope = _read(OpenCvEvaluationFrameReader(cv2_module=fake_cv2), tmp_path, _image_frame())

    assert fake_cv2.imread_calls == [(str(media.resolve()), fake_cv2.IMREAD_COLOR)]
    assert envelope.width == 2
    assert envelope.height == 2
    assert envelope.payload == image.tobytes(order="C")
    image.fill(0)
    assert envelope.payload != image.tobytes(order="C")


def test_reader_seeks_and_verifies_exact_video_frame(tmp_path: Path) -> None:
    media = tmp_path / "media" / "clip.mp4"
    media.parent.mkdir()
    media.write_bytes(b"video")
    capture = FakeCapture(np.full((2, 2, 3), 9, dtype=np.uint8), position=8.0)
    fake_cv2 = FakeCv2(capture=capture)

    envelope = _read(OpenCvEvaluationFrameReader(cv2_module=fake_cv2), tmp_path, _video_frame())

    assert capture.set_calls == [(fake_cv2.CAP_PROP_POS_FRAMES, 7.0)]
    assert capture.released is True
    assert envelope.sequence_number == 7
    assert envelope.payload == bytes([9]) * 12


def test_reader_rejects_approximate_video_seek_and_releases_capture(tmp_path: Path) -> None:
    media = tmp_path / "media" / "clip.mp4"
    media.parent.mkdir()
    media.write_bytes(b"video")
    capture = FakeCapture(np.zeros((2, 2, 3), dtype=np.uint8), position=6.0)

    with pytest.raises(EvaluationRuntimeError, match="exact requested video frame"):
        _read(
            OpenCvEvaluationFrameReader(cv2_module=FakeCv2(capture=capture)),
            tmp_path,
            _video_frame(),
        )

    assert capture.released is True


def test_reader_rejects_dimension_mismatch_and_path_escape(tmp_path: Path) -> None:
    media = tmp_path / "media" / "frame.jpg"
    media.parent.mkdir()
    media.write_bytes(b"image")
    reader = OpenCvEvaluationFrameReader(
        cv2_module=FakeCv2(image=np.zeros((3, 2, 3), dtype=np.uint8))
    )
    with pytest.raises(EvaluationRuntimeError, match="dimensions"):
        _read(reader, tmp_path, _image_frame())

    escaped = _image_frame().model_copy(update={"media_path": "../outside.jpg"})
    with pytest.raises(EvaluationRuntimeError, match="outside dataset root"):
        _read(reader, tmp_path, escaped)


def test_reader_rejects_media_symlink_before_resolving_it(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    target = media_dir / "real.jpg"
    target.write_bytes(b"image")
    link = media_dir / "frame.jpg"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    reader = OpenCvEvaluationFrameReader(
        cv2_module=FakeCv2(image=np.zeros((2, 2, 3), dtype=np.uint8))
    )

    with pytest.raises(EvaluationRuntimeError, match="symlink"):
        _read(reader, tmp_path, _image_frame())


def test_evaluation_media_exposes_envelope_and_drawable_image(tmp_path: Path) -> None:
    media_path = tmp_path / "media" / "frame.jpg"
    media_path.parent.mkdir()
    media_path.write_bytes(b"image")
    image = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
    media = OpenCvEvaluationMedia(cv2_module=FakeCv2(image=image))

    envelope, drawable = media.load(
        _image_frame(),
        dataset_root=tmp_path,
        camera_external_id="evaluation-camera",
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
        sequence_number=3,
    )

    assert envelope.stream_id == "evaluation:media/frame.jpg"
    assert envelope.sequence_number == 3
    assert np.array_equal(drawable, image)
    assert drawable is not image
    media.close()


def _envelope() -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="evaluation:test",
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
        camera_external_id="evaluation-camera",
        captured_at=datetime(2026, 9, 24, tzinfo=UTC),
        width=10,
        height=10,
        sequence_number=1,
        payload=bytes(10 * 10 * 3),
    )


def _overlay_plan() -> OverlayPlan:
    box = OverlayBoxInstruction.model_validate(
        {
            "source": "PREDICTION",
            "object_id": "ppe-1",
            "class_name": "NO-Hardhat",
            "bounding_box": {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6},
            "match_status": "FP",
            "color": {"red": 10, "green": 20, "blue": 30},
            "label": "PRED | NO-Hardhat | FP",
            "related_person_object_id": "person-1",
        }
    )
    person = box.model_copy(
        update={
            "object_id": "person-1",
            "class_name": "Person",
            "bounding_box": box.bounding_box.model_copy(
                update={"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
            ),
            "related_person_object_id": None,
        }
    )
    relation = OverlayRelationInstruction(
        source_key="PREDICTION:ppe-1", target_key="PREDICTION:person-1"
    )
    return OverlayPlan(boxes=(box, person), relations=(relation,))


def test_renderer_draws_boxes_relations_and_legend_without_mutating_input() -> None:
    fake_cv2 = FakeCv2()
    original = _envelope()

    rendered = OpenCvOverlayRenderer(cv2_module=fake_cv2).render(original, _overlay_plan())

    assert len(fake_cv2.rectangle_calls) == 2
    assert len(fake_cv2.line_calls) == 1
    assert len(fake_cv2.text_calls) >= 2 + len(_overlay_plan().legend)
    assert fake_cv2.rectangle_calls[0][3] == (30, 20, 10)  # RGB plan -> OpenCV BGR
    assert original.payload == bytes(10 * 10 * 3)
    assert rendered is not original
    assert rendered.model_dump(exclude={"payload"}) == original.model_dump(exclude={"payload"})


def test_renderer_writes_only_supported_image_output(tmp_path: Path) -> None:
    fake_cv2 = FakeCv2()
    renderer = OpenCvOverlayRenderer(cv2_module=fake_cv2)
    output = tmp_path / "nested" / "annotated.png"

    renderer.write(output, _envelope())

    assert output.parent.is_dir()
    assert fake_cv2.imwrite_calls[0][0] == str(output.resolve())
    with pytest.raises(EvaluationRuntimeError, match="image extension"):
        renderer.write(tmp_path / "evidence.mp4", _envelope())


def test_evidence_exporter_renders_stable_frame_filename_and_closes(tmp_path: Path) -> None:
    fake_cv2 = FakeCv2()
    exporter = OpenCvEvidenceExporter(tmp_path, cv2_module=fake_cv2)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    output = exporter.write("frame-01", image, _overlay_plan())

    assert output == (tmp_path / "frame-01.png").resolve()
    assert fake_cv2.imwrite_calls[0][0] == str(output)
    exporter.close()


def test_evidence_exporter_rejects_unsafe_frame_id(tmp_path: Path) -> None:
    exporter = OpenCvEvidenceExporter(tmp_path, cv2_module=FakeCv2())
    with pytest.raises(EvaluationRuntimeError, match="frame_id"):
        exporter.write("../escape", np.zeros((10, 10, 3), dtype=np.uint8), _overlay_plan())


def _validation_arguments(tmp_path: Path) -> ProviderValidationArguments:
    model = tmp_path / "model.pt"
    data = tmp_path / "data.yaml"
    model.write_bytes(b"weights")
    data.write_text("path: .", encoding="utf-8")
    return ProviderValidationArguments(
        model_path=model.resolve(),
        data_config_path=data.resolve(),
        split="test",
        image_size=(640, 480),
        confidence_threshold=0.001,
        iou_threshold=0.7,
        max_detections=300,
        batch_size=4,
        workers=0,
        device="cpu",
        seed=7,
    )


def test_ultralytics_provider_uses_exact_local_inputs_and_locked_arguments(tmp_path: Path) -> None:
    arguments = _validation_arguments(tmp_path)
    calls: list[Path] = []

    class FakeModel:
        names = {0: "Person", 1: "Hardhat", 2: "NO-Hardhat", 3: "Safety Vest", 4: "NO-Safety Vest"}

        def val(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.5))

    model = FakeModel()

    def factory(path: Path) -> FakeModel:
        calls.append(path)
        return model

    result = UltralyticsValidationProvider(
        model_factory=factory,
        provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
    ).validate(arguments)

    assert calls == [arguments.model_path]
    assert model.kwargs == {
        "data": str(arguments.data_config_path),
        "split": "test",
        "imgsz": (480, 640),
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "batch": 4,
        "workers": 0,
        "device": "cpu",
        "seed": 7,
        "deterministic": True,
        "plots": False,
        "save_json": False,
        "verbose": False,
    }
    assert result["provider_version"] == PINNED_ULTRALYTICS_VERSION
    assert result["ap50"] == pytest.approx(0.8)
    assert result["ap50_95"] == pytest.approx(0.5)


def test_ultralytics_provider_fails_before_loading_on_wrong_version_or_missing_file(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    loaded = False

    def factory(path: Path) -> object:
        nonlocal loaded
        del path
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="pinned"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: "8.4.154",
        ).validate(arguments)
    assert loaded is False

    arguments.model_path.unlink()
    with pytest.raises(EvaluationRuntimeError, match="regular local file"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)
    assert loaded is False
