"""Explicit runtime adapters for offline, reproducible evaluation.

Importing this module does not import OpenCV, NumPy, Torch, or Ultralytics.
Those providers are loaded only when a concrete adapter is constructed or used.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

from smartsite_ai.evaluation.models import EvaluationFrame
from smartsite_ai.evaluation.overlay import OverlayColor, OverlayPlan
from smartsite_ai.evaluation.provider_validation import (
    PINNED_ULTRALYTICS_VERSION,
    ProviderValidationArguments,
)
from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP
from smartsite_ai.ingestion.envelope import FrameEnvelope

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
_SAFE_FRAME_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVALUATION_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


class EvaluationRuntimeError(RuntimeError):
    """Safe failure raised by a concrete evaluation runtime adapter."""


def _load_cv2() -> object:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on selected runtime extra
        raise EvaluationRuntimeError("OpenCV runtime is unavailable") from error
    return cv2


def _load_numpy() -> object:
    try:
        import numpy
    except ImportError as error:  # pragma: no cover - installed with the vision extra
        raise EvaluationRuntimeError("NumPy runtime is unavailable") from error
    return numpy


def _resolved_media_path(dataset_root: Path, media_path: str) -> Path:
    root = dataset_root.resolve()
    windows_path = PureWindowsPath(media_path)
    posix_path = PurePosixPath(media_path.replace("\\", "/"))
    if (
        windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    ):
        raise EvaluationRuntimeError("evaluation media path resolves outside dataset root")
    unresolved_path = root
    for part in posix_path.parts:
        unresolved_path /= part
        if unresolved_path.is_symlink():
            raise EvaluationRuntimeError("evaluation media path must not contain a symlink")
    path = unresolved_path.resolve()
    if not path.is_relative_to(root):
        raise EvaluationRuntimeError("evaluation media path resolves outside dataset root")
    if not path.is_file():
        raise EvaluationRuntimeError("evaluation media path must be an existing regular file")
    return path


def _validated_bgr_image(image: object, frame: EvaluationFrame) -> object:
    shape = getattr(image, "shape", None)
    dtype = getattr(image, "dtype", None)
    if shape != (frame.height, frame.width, 3):
        raise EvaluationRuntimeError("decoded media dimensions do not match evaluation frame")
    if str(dtype) != "uint8":
        raise EvaluationRuntimeError("decoded media must contain uint8 BGR pixels")
    return image


class OpenCvEvaluationFrameReader:
    """Read one validated image or exact indexed video frame into packed BGR24."""

    def __init__(self, *, cv2_module: object | None = None) -> None:
        self._cv2 = cv2_module or _load_cv2()

    def read(
        self,
        dataset_root: Path,
        frame: EvaluationFrame,
        *,
        stream_id: str,
        session_id: UUID,
        camera_external_id: str,
        captured_at: datetime,
        sequence_number: int,
    ) -> FrameEnvelope:
        image = self.read_image(dataset_root, frame)
        return self.envelope(
            image,
            stream_id=stream_id,
            session_id=session_id,
            camera_external_id=camera_external_id,
            captured_at=captured_at,
            sequence_number=sequence_number,
        )

    def read_image(self, dataset_root: Path, frame: EvaluationFrame) -> object:
        path = _resolved_media_path(dataset_root, frame.media_path)
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            image = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
            if image is None:
                raise EvaluationRuntimeError("OpenCV could not decode evaluation image")
            return _validated_bgr_image(image, frame)
        if suffix not in _VIDEO_SUFFIXES or frame.frame_index is None:
            raise EvaluationRuntimeError("evaluation frame does not describe supported media")
        return self._read_video_frame(path, frame)

    def _read_video_frame(self, path: Path, frame: EvaluationFrame) -> object:
        capture = self._cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise EvaluationRuntimeError("OpenCV could not open evaluation video")
            requested_index = frame.frame_index
            assert requested_index is not None
            if not capture.set(self._cv2.CAP_PROP_POS_FRAMES, float(requested_index)):
                raise EvaluationRuntimeError("OpenCV could not seek to requested video frame")
            ok, image = capture.read()
            if not ok or image is None:
                raise EvaluationRuntimeError("OpenCV could not decode requested video frame")
            position_after_read = float(capture.get(self._cv2.CAP_PROP_POS_FRAMES))
            if not math.isfinite(position_after_read) or not math.isclose(
                position_after_read,
                float(requested_index + 1),
                abs_tol=0.01,
            ):
                raise EvaluationRuntimeError(
                    "OpenCV did not return the exact requested video frame"
                )
            return _validated_bgr_image(image, frame)
        finally:
            capture.release()

    @staticmethod
    def envelope(
        image: object,
        *,
        stream_id: str,
        session_id: UUID,
        camera_external_id: str,
        captured_at: datetime,
        sequence_number: int,
    ) -> FrameEnvelope:
        shape = getattr(image, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3:
            raise EvaluationRuntimeError("decoded image has an invalid BGR shape")
        numpy = _load_numpy()
        packed = numpy.ascontiguousarray(image)
        height, width, channels = packed.shape
        if channels != 3 or str(packed.dtype) != "uint8":
            raise EvaluationRuntimeError("decoded image must be packed uint8 BGR24")
        return FrameEnvelope(
            stream_id=stream_id,
            session_id=session_id,
            camera_external_id=camera_external_id,
            captured_at=captured_at,
            width=width,
            height=height,
            sequence_number=sequence_number,
            payload=packed.tobytes(order="C"),
        )


class OpenCvEvaluationMedia:
    """Orchestration-facing reader returning both detector and drawing representations."""

    def __init__(self, *, cv2_module: object | None = None) -> None:
        self._reader = OpenCvEvaluationFrameReader(cv2_module=cv2_module)

    def load(
        self,
        frame: EvaluationFrame,
        *,
        dataset_root: Path,
        camera_external_id: str,
        session_id: UUID,
        sequence_number: int,
    ) -> tuple[FrameEnvelope, object]:
        image = self._reader.read_image(dataset_root, frame)
        numpy = _load_numpy()
        drawable = numpy.ascontiguousarray(image).copy()
        captured_at = _EVALUATION_EPOCH + timedelta(seconds=frame.video_time_seconds or 0.0)
        stream_id = f"evaluation:{PurePosixPath(frame.media_path.replace('\\', '/')).as_posix()}"
        envelope = self._reader.envelope(
            image,
            stream_id=stream_id,
            session_id=session_id,
            camera_external_id=camera_external_id,
            captured_at=captured_at,
            sequence_number=sequence_number,
        )
        return envelope, drawable

    def close(self) -> None:
        """Release adapter state; per-read video captures are already closed."""


def _bgr(color: OverlayColor) -> tuple[int, int, int]:
    return (color.blue, color.green, color.red)


class OpenCvOverlayRenderer:
    """Render deterministic OverlayPlan instructions into owned BGR images."""

    def __init__(self, *, cv2_module: object | None = None) -> None:
        self._cv2 = cv2_module or _load_cv2()

    def render_image(self, image: object, plan: OverlayPlan) -> object:
        numpy = _load_numpy()
        rendered = numpy.ascontiguousarray(image).copy()
        if len(rendered.shape) != 3 or rendered.shape[2] != 3 or str(rendered.dtype) != "uint8":
            raise EvaluationRuntimeError("overlay input must be a uint8 BGR image")
        height, width, _ = rendered.shape
        centers: dict[str, tuple[int, int]] = {}
        for box in plan.boxes:
            x1 = min(width - 1, max(0, round(box.bounding_box.x1 * (width - 1))))
            y1 = min(height - 1, max(0, round(box.bounding_box.y1 * (height - 1))))
            x2 = min(width - 1, max(0, round(box.bounding_box.x2 * (width - 1))))
            y2 = min(height - 1, max(0, round(box.bounding_box.y2 * (height - 1))))
            color = _bgr(box.color)
            self._cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            self._cv2.putText(
                rendered,
                box.label,
                (x1, max(0, y1 - 4)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                self._cv2.LINE_AA,
            )
            centers[f"{box.source}:{box.object_id}"] = ((x1 + x2) // 2, (y1 + y2) // 2)
        for relation in plan.relations:
            source = centers.get(relation.source_key)
            target = centers.get(relation.target_key)
            if source is None or target is None:
                raise EvaluationRuntimeError("overlay relation references an unknown rendered box")
            self._cv2.line(rendered, source, target, _bgr(relation.color), 1, self._cv2.LINE_AA)
        legend_y = 16
        for entry in plan.legend:
            self._cv2.putText(
                rendered,
                entry.label,
                (4, legend_y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                _bgr(entry.color),
                1,
                self._cv2.LINE_AA,
            )
            legend_y += 14
        return rendered

    def render(self, frame: FrameEnvelope, plan: OverlayPlan) -> FrameEnvelope:
        numpy = _load_numpy()
        image = numpy.frombuffer(frame.payload, dtype=numpy.uint8).reshape(
            frame.height, frame.width, 3
        )
        rendered = self.render_image(image, plan)
        return frame.model_copy(update={"payload": rendered.tobytes(order="C")})

    def write(self, path: Path, frame: FrameEnvelope) -> None:
        suffix = path.suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            raise EvaluationRuntimeError("annotated evidence output requires an image extension")
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        numpy = _load_numpy()
        image = numpy.frombuffer(frame.payload, dtype=numpy.uint8).reshape(
            frame.height, frame.width, 3
        )
        if not self._cv2.imwrite(str(resolved), image):
            raise EvaluationRuntimeError("OpenCV could not write annotated evidence image")


class OpenCvEvidenceExporter:
    """Write one stable PNG per frame after applying an OverlayPlan."""

    def __init__(self, output_dir: Path, *, cv2_module: object | None = None) -> None:
        self._output_dir = output_dir.resolve()
        self._renderer = OpenCvOverlayRenderer(cv2_module=cv2_module)

    def write(self, frame_id: str, image: object, plan: OverlayPlan) -> Path:
        if _SAFE_FRAME_ID.fullmatch(frame_id) is None:
            raise EvaluationRuntimeError("frame_id is unsafe for an evidence filename")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output = (self._output_dir / f"{frame_id}.png").resolve()
        if not output.is_relative_to(self._output_dir):
            raise EvaluationRuntimeError("frame_id is unsafe for an evidence filename")
        rendered = self._renderer.render_image(image, plan)
        if not self._renderer._cv2.imwrite(str(output), rendered):
            raise EvaluationRuntimeError("OpenCV could not write annotated evidence image")
        return output

    def close(self) -> None:
        """Release adapter state; OpenCV image writing holds no persistent handle."""


ModelFactory = Callable[[Path], object]
ProviderVersionFactory = Callable[[], str]


def _ultralytics_model(path: Path) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depends on selected runtime extra
        raise EvaluationRuntimeError("Ultralytics runtime is unavailable") from error
    return YOLO(str(path), task="detect")


def _ultralytics_version() -> str:
    return distribution_version("ultralytics")


def _require_regular_local_file(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EvaluationRuntimeError(f"{label} must be an absolute regular local file")


def _reject_downloadable_data_config(path: Path) -> None:
    try:
        if path.stat().st_size > 256 * 1024:
            raise EvaluationRuntimeError("provider data config exceeds 256 KiB")
        text = path.read_text(encoding="utf-8")
    except EvaluationRuntimeError:
        raise
    except (OSError, UnicodeError) as error:
        raise EvaluationRuntimeError("provider data config cannot be read as UTF-8") from error
    lowered = text.lower()
    if "download:" in lowered or "http://" in lowered or "https://" in lowered:
        raise EvaluationRuntimeError("provider data config may not declare downloads or URLs")


class UltralyticsValidationProvider:
    """Pinned, local-only Ultralytics implementation of ValidationProvider."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        provider_version_factory: ProviderVersionFactory | None = None,
    ) -> None:
        self._model_factory = model_factory or _ultralytics_model
        self._provider_version_factory = provider_version_factory or _ultralytics_version

    def validate(self, arguments: ProviderValidationArguments) -> object:
        provider_version = self._provider_version_factory()
        if provider_version != PINNED_ULTRALYTICS_VERSION:
            raise EvaluationRuntimeError(
                f"Ultralytics runtime must equal pinned version {PINNED_ULTRALYTICS_VERSION}"
            )
        _require_regular_local_file(arguments.model_path, "model_path")
        _require_regular_local_file(arguments.data_config_path, "data_config_path")
        _reject_downloadable_data_config(arguments.data_config_path)
        model = self._model_factory(arguments.model_path)
        class_map = getattr(model, "names", None)
        if not isinstance(class_map, Mapping) or tuple(sorted(class_map.items())) != (
            CANONICAL_PPE_CLASS_MAP
        ):
            raise EvaluationRuntimeError("Ultralytics model class map is not canonical PPE")
        metrics = model.val(
            data=str(arguments.data_config_path),
            split=arguments.split,
            imgsz=(arguments.image_size[1], arguments.image_size[0]),
            conf=arguments.confidence_threshold,
            iou=arguments.iou_threshold,
            max_det=arguments.max_detections,
            batch=arguments.batch_size,
            workers=arguments.workers,
            device=arguments.device,
            seed=arguments.seed,
            deterministic=arguments.deterministic,
            plots=arguments.plots,
            save_json=arguments.save_json,
            verbose=False,
        )
        box = getattr(metrics, "box", None)
        ap50 = getattr(box, "map50", None)
        ap50_95 = getattr(box, "map", None)
        if (
            isinstance(ap50, bool)
            or not isinstance(ap50, (int, float))
            or isinstance(ap50_95, bool)
            or not isinstance(ap50_95, (int, float))
        ):
            raise EvaluationRuntimeError("Ultralytics validation returned invalid AP metrics")
        return {
            "provider_name": "ultralytics",
            "provider_version": provider_version,
            "class_map": CANONICAL_PPE_CLASS_MAP,
            "ap50": float(ap50),
            "ap50_95": float(ap50_95),
        }


__all__ = [
    "EvaluationRuntimeError",
    "OpenCvEvaluationFrameReader",
    "OpenCvEvaluationMedia",
    "OpenCvEvidenceExporter",
    "OpenCvOverlayRenderer",
    "UltralyticsValidationProvider",
]
