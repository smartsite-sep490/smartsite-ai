import os
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
    (tmp_path / "test" / "images").mkdir(parents=True)
    (tmp_path / "test" / "labels").mkdir()
    model.write_bytes(b"weights")
    data.write_text(
        f"path: {tmp_path.resolve().as_posix()}\n"
        "train: test/images\n"
        "val: test/images\n"
        "test: test/images\n",
        encoding="utf-8",
    )
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


def test_real_ultralytics_import_and_font_lookup_stay_in_runtime_sandbox(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-config"
    outside.mkdir()
    source_root = Path(__file__).parents[2] / "src"
    code = """
from pathlib import Path
import tempfile
from smartsite_ai.evaluation.runtime_adapters import (
    _temporary_validation_workspace,
    ultralytics_runtime_sandbox,
)

original_temp = tempfile.gettempdir()
with ultralytics_runtime_sandbox() as root:
    import ultralytics.utils as ultralytics_utils
    from ultralytics.utils.checks import check_font

    config = Path(ultralytics_utils.USER_CONFIG_DIR).resolve()
    settings = Path(ultralytics_utils.SETTINGS_FILE).resolve()
    config.relative_to(root)
    settings.relative_to(root)
    font = Path(check_font("Arial.ttf")).resolve()
    font.relative_to(root)
    assert font.is_file()
    configured_temp = Path(tempfile.gettempdir()).resolve()
    configured_temp.relative_to(root)
    with tempfile.NamedTemporaryFile() as handle:
        Path(handle.name).resolve().relative_to(root)
    workspace = _temporary_validation_workspace()
    Path(workspace.name).resolve().relative_to(root)
    workspace.cleanup()
    print(f"SANDBOX={root}")
assert not root.exists()
assert tempfile.gettempdir() == original_temp
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    environment["YOLO_CONFIG_DIR"] = str(outside)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SANDBOX=" in completed.stdout
    assert list(outside.iterdir()) == []


def test_real_ultralytics_provider_refuses_to_import_without_runtime_sandbox(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)

    with pytest.raises(EvaluationRuntimeError, match="requires an active runtime sandbox"):
        UltralyticsValidationProvider(
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)


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
    project = Path(str(model.kwargs.pop("project")))
    assert not project.exists()
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
        "name": "validation",
        "exist_ok": False,
        "save": False,
        "plots": False,
        "save_json": False,
        "verbose": False,
    }
    assert result["provider_version"] == PINNED_ULTRALYTICS_VERSION
    assert result["ap50"] == pytest.approx(0.8)
    assert result["ap50_95"] == pytest.approx(0.5)


@pytest.mark.parametrize("provider_fails", [False, True])
def test_ultralytics_provider_isolates_outputs_and_removes_only_new_caches(
    tmp_path: Path,
    provider_fails: bool,
) -> None:
    arguments = _validation_arguments(tmp_path)
    pre_existing = tmp_path / "val" / "labels.cache"
    pre_existing.parent.mkdir()
    pre_existing.write_bytes(b"keep this cache")
    generated = tmp_path / "test" / "labels.cache"
    workspaces: list[Path] = []

    class CacheCreatingModel:
        names = {
            0: "Person",
            1: "Hardhat",
            2: "NO-Hardhat",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }

        def val(self, **kwargs: object) -> object:
            project = Path(str(kwargs["project"]))
            workspaces.append(project)
            output = project / str(kwargs["name"])
            output.mkdir()
            (output / "results.csv").write_text("provider output", encoding="utf-8")
            generated.write_bytes(b"generated cache")
            if provider_fails:
                raise RuntimeError("provider failed")
            return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.5))

    provider = UltralyticsValidationProvider(
        model_factory=lambda _path: CacheCreatingModel(),
        provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
    )

    if provider_fails:
        with pytest.raises(RuntimeError, match="provider failed"):
            provider.validate(arguments)
    else:
        provider.validate(arguments)

    assert pre_existing.read_bytes() == b"keep this cache"
    assert not generated.exists()
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_ultralytics_provider_discards_result_when_existing_cache_changes(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    pre_existing = tmp_path / "test" / "labels.cache"
    pre_existing.write_bytes(b"original cache")

    class CacheMutatingModel:
        names = {
            0: "Person",
            1: "Hardhat",
            2: "NO-Hardhat",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }

        def val(self, **_kwargs: object) -> object:
            pre_existing.write_bytes(b"mutated cache")
            return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.5))

    provider = UltralyticsValidationProvider(
        model_factory=lambda _path: CacheMutatingModel(),
        provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
    )

    with pytest.raises(EvaluationRuntimeError, match="cleanup failed"):
        provider.validate(arguments)

    assert pre_existing.exists()


def test_ultralytics_provider_fails_safely_when_generated_cache_cannot_be_removed(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    generated = tmp_path / "test" / "labels.cache"

    class CacheCreatingModel:
        names = {
            0: "Person",
            1: "Hardhat",
            2: "NO-Hardhat",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }

        def val(self, **_kwargs: object) -> object:
            generated.write_bytes(b"generated cache")
            return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.5))

    def reject_unlink(_path: Path) -> None:
        raise PermissionError("locked")

    provider = UltralyticsValidationProvider(
        model_factory=lambda _path: CacheCreatingModel(),
        provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        cache_unlink=reject_unlink,
    )

    with pytest.raises(EvaluationRuntimeError, match="cleanup failed"):
        provider.validate(arguments)

    assert generated.exists()


def test_ultralytics_provider_discards_result_when_workspace_cleanup_fails(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    workspace_path = tmp_path / "isolated-output"

    class FailingWorkspace:
        name = str(workspace_path.resolve())

        def __init__(self) -> None:
            workspace_path.mkdir()

        def cleanup(self) -> None:
            raise PermissionError("locked")

    class SuccessfulModel:
        names = {
            0: "Person",
            1: "Hardhat",
            2: "NO-Hardhat",
            3: "Safety Vest",
            4: "NO-Safety Vest",
        }

        def val(self, **_kwargs: object) -> object:
            return SimpleNamespace(box=SimpleNamespace(map50=0.8, map=0.5))

    provider = UltralyticsValidationProvider(
        model_factory=lambda _path: SuccessfulModel(),
        provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        workspace_factory=FailingWorkspace,
    )

    with pytest.raises(EvaluationRuntimeError, match="cleanup failed"):
        provider.validate(arguments)

    assert workspace_path.exists()


def test_ultralytics_provider_rejects_split_outside_declared_dataset_root(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    arguments.data_config_path.write_text(
        f"path: {tmp_path.as_posix()}\ntest: ../{outside.name}\n",
        encoding="utf-8",
    )
    loaded = False

    def factory(_path: Path) -> object:
        nonlocal loaded
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="inside the declared dataset root"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)

    assert loaded is False


def test_ultralytics_provider_rejects_unselected_split_outside_dataset_root(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-train"
    outside_images = outside / "images"
    outside_labels = outside / "labels"
    outside_images.mkdir(parents=True)
    outside_labels.mkdir()
    arguments.data_config_path.write_text(
        f"path: {tmp_path.as_posix()}\n"
        f"train: ../{outside.name}/images\n"
        "val: test/images\n"
        "test: test/images\n",
        encoding="utf-8",
    )
    loaded = False

    def factory(_path: Path) -> object:
        nonlocal loaded
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="inside the declared dataset root"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)

    assert loaded is False


def test_ultralytics_provider_rejects_split_list_files_that_can_escape_root(
    tmp_path: Path,
) -> None:
    arguments = _validation_arguments(tmp_path)
    split_index = tmp_path / "test.txt"
    split_index.write_text("../outside/image.jpg\n", encoding="utf-8")
    arguments.data_config_path.write_text(
        f"path: {tmp_path.resolve().as_posix()}\ntest: test.txt\n", encoding="utf-8"
    )
    loaded = False

    def factory(_path: Path) -> object:
        nonlocal loaded
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="split paths must be directories"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)

    assert loaded is False


def test_ultralytics_provider_rejects_relative_dataset_root_before_model_load(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configuration"
    dataset_root = tmp_path / "dataset"
    config_root.mkdir()
    (dataset_root / "test" / "images").mkdir(parents=True)
    (dataset_root / "test" / "labels").mkdir()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    data_path = config_root / "data.yaml"
    data_path.write_text("path: ../dataset\ntest: test/images\n", encoding="utf-8")
    arguments = _validation_arguments(tmp_path).model_copy(
        update={"model_path": model_path.resolve(), "data_config_path": data_path.resolve()}
    )
    loaded = False

    def factory(_path: Path) -> object:
        nonlocal loaded
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="path must be absolute"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)

    assert loaded is False


def test_ultralytics_provider_rejects_images_root_with_labels_outside_dataset_root(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "labels").mkdir()
    arguments = _validation_arguments(tmp_path).model_copy(
        update={"data_config_path": (tmp_path / "images-root.yaml").resolve()}
    )
    arguments.data_config_path.write_text(
        f"path: {images.resolve().as_posix()}\ntest: .\n", encoding="utf-8"
    )
    loaded = False

    def factory(_path: Path) -> object:
        nonlocal loaded
        loaded = True
        return object()

    with pytest.raises(EvaluationRuntimeError, match="label and cache paths must remain inside"):
        UltralyticsValidationProvider(
            model_factory=factory,
            provider_version_factory=lambda: PINNED_ULTRALYTICS_VERSION,
        ).validate(arguments)

    assert loaded is False


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
