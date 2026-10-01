"""Explicit runtime adapters for offline, reproducible evaluation.

Importing this module does not import OpenCV, NumPy, Torch, or Ultralytics.
Those providers are loaded only when a concrete adapter is constructed or used.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol
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
_MAX_CACHE_DIRECTORIES = 4_096
_MAX_CACHE_FILES = 1_024
_MAX_CACHE_SNAPSHOT_BYTES = 128 * 1024 * 1024
_ULTRALYTICS_SANDBOX_ENV = "SMARTSITE_ULTRALYTICS_SANDBOX_ROOT"
_SANDBOX_ENVIRONMENT = (
    _ULTRALYTICS_SANDBOX_ENV,
    "HF_HOME",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "PYTHONDONTWRITEBYTECODE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TORCH_HOME",
    "WANDB_DISABLED",
    "YOLO_AUTOINSTALL",
    "YOLO_CONFIG_DIR",
    "YOLO_OFFLINE",
)
_active_ultralytics_sandbox_root: Path | None = None


class EvaluationRuntimeError(RuntimeError):
    """Safe failure raised by a concrete evaluation runtime adapter."""


def _matplotlib_font_source() -> Path | None:
    spec = find_spec("matplotlib")
    if spec is None or spec.origin is None:
        return None
    font = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    if font.is_symlink() or not font.is_file():
        raise EvaluationRuntimeError("bundled evaluation font is unavailable")
    return font


def _require_ultralytics_runtime_sandbox() -> Path:
    raw_root = os.environ.get(_ULTRALYTICS_SANDBOX_ENV)
    if not raw_root:
        raise EvaluationRuntimeError("Ultralytics validation requires an active runtime sandbox")
    try:
        root = Path(raw_root).resolve(strict=True)
        config = Path(os.environ["YOLO_CONFIG_DIR"]).resolve(strict=True)
        matplotlib = Path(os.environ["MPLCONFIGDIR"]).resolve(strict=True)
        config.relative_to(root)
        matplotlib.relative_to(root)
    except (KeyError, OSError, ValueError) as error:
        raise EvaluationRuntimeError("Ultralytics runtime sandbox is invalid") from error
    if os.environ.get("YOLO_OFFLINE", "").casefold() != "true":
        raise EvaluationRuntimeError("Ultralytics runtime sandbox must enforce offline mode")
    if os.environ.get("YOLO_AUTOINSTALL", "").casefold() != "false":
        raise EvaluationRuntimeError("Ultralytics runtime sandbox must disable auto-install")
    if _active_ultralytics_sandbox_root != root:
        raise EvaluationRuntimeError("Ultralytics runtime sandbox is not owned by this process")
    return root


@contextmanager
def ultralytics_runtime_sandbox() -> Iterator[Path]:
    """Contain evaluator provider configuration, caches, temporary files, and fonts."""

    global _active_ultralytics_sandbox_root
    if _active_ultralytics_sandbox_root is not None:
        raise EvaluationRuntimeError("Ultralytics runtime sandbox may not be nested")
    if any(name == "ultralytics" or name.startswith("ultralytics.") for name in sys.modules):
        raise EvaluationRuntimeError(
            "Ultralytics runtime sandbox must start before importing the provider"
        )
    previous = {name: os.environ.get(name) for name in _SANDBOX_ENVIRONMENT}
    previous_tempdir = tempfile.tempdir
    try:
        with tempfile.TemporaryDirectory(prefix="smartsite-ultralytics-runtime-") as raw_root:
            root = Path(raw_root).resolve()
            config = root / "config"
            user_config = config / "Ultralytics"
            matplotlib = root / "matplotlib"
            temporary = root / "tmp"
            for directory in (user_config, matplotlib, temporary):
                directory.mkdir(parents=True)
            font = _matplotlib_font_source()
            for name in ("Arial.ttf", "Arial.Unicode.ttf"):
                destination = user_config / name
                if font is None:
                    destination.touch()
                else:
                    shutil.copyfile(font, destination)
            environment = {
                _ULTRALYTICS_SANDBOX_ENV: str(root),
                "HF_HOME": str(root / "huggingface"),
                "MPLCONFIGDIR": str(matplotlib),
                "NUMBA_CACHE_DIR": str(root / "numba"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TEMP": str(temporary),
                "TMP": str(temporary),
                "TMPDIR": str(temporary),
                "TORCH_HOME": str(root / "torch"),
                "WANDB_DISABLED": "true",
                "YOLO_AUTOINSTALL": "false",
                "YOLO_CONFIG_DIR": str(config),
                "YOLO_OFFLINE": "true",
            }
            os.environ.update(environment)
            tempfile.tempdir = str(temporary)
            _active_ultralytics_sandbox_root = root
            try:
                yield root
            finally:
                _active_ultralytics_sandbox_root = None
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
    sandbox_root = _require_ultralytics_runtime_sandbox()
    try:
        import ultralytics.utils as ultralytics_utils
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depends on selected runtime extra
        raise EvaluationRuntimeError("Ultralytics runtime is unavailable") from error
    try:
        Path(ultralytics_utils.USER_CONFIG_DIR).resolve(strict=True).relative_to(sandbox_root)
        Path(ultralytics_utils.SETTINGS_FILE).resolve(strict=True).relative_to(sandbox_root)
    except (OSError, ValueError) as error:
        raise EvaluationRuntimeError(
            "Ultralytics provider configuration escaped the runtime sandbox"
        ) from error
    return YOLO(str(path), task="detect")


def _ultralytics_version() -> str:
    return distribution_version("ultralytics")


def _require_regular_local_file(path: Path, label: str) -> None:
    if not path.is_absolute() or _is_link_or_junction(path) or not path.is_file():
        raise EvaluationRuntimeError(f"{label} must be an absolute regular local file")


def _read_provider_data_config(path: Path) -> Mapping[object, object]:
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
    try:
        import yaml

        document = yaml.safe_load(text)
    except ImportError as error:  # pragma: no cover - installed with the vision extra
        raise EvaluationRuntimeError("PyYAML runtime is unavailable") from error
    except Exception as error:
        raise EvaluationRuntimeError("provider data config must be valid YAML") from error
    if not isinstance(document, Mapping):
        raise EvaluationRuntimeError("provider data config root must be a mapping")
    return document


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _require_non_symlink_path(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current /= part
            if _is_link_or_junction(current):
                raise EvaluationRuntimeError(f"{label} must not traverse a symlink or junction")
    except OSError as error:
        raise EvaluationRuntimeError(f"{label} cannot be inspected safely") from error


def _provider_dataset_root(
    data_config_path: Path,
    document: Mapping[object, object],
    split: str,
) -> Path:
    raw_root = document.get("path", ".")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise EvaluationRuntimeError("provider data config path must be a non-blank string")
    configured_root = Path(raw_root)
    if not configured_root.is_absolute():
        raise EvaluationRuntimeError("provider data config path must be absolute")
    _require_non_symlink_path(configured_root, "provider dataset root")
    try:
        root = configured_root.resolve(strict=True)
    except OSError as error:
        raise EvaluationRuntimeError("provider dataset root must exist") from error
    if not root.is_dir() or root == Path(root.anchor):
        raise EvaluationRuntimeError("provider dataset root must be a bounded local directory")

    for split_name in dict.fromkeys((split, "train", "val")):
        raw_split = document.get(split_name)
        split_paths = [raw_split] if isinstance(raw_split, str) else raw_split
        if not isinstance(split_paths, list) or not split_paths:
            raise EvaluationRuntimeError(
                f"provider data config must declare the {split_name} split"
            )
        for raw_path in split_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise EvaluationRuntimeError("provider split paths must be non-blank strings")
            configured_path = Path(raw_path)
            if not configured_path.is_absolute():
                configured_path = root / configured_path
            _require_non_symlink_path(configured_path, "provider split path")
            try:
                resolved_path = configured_path.resolve(strict=True)
                resolved_path.relative_to(root)
            except (OSError, ValueError) as error:
                raise EvaluationRuntimeError(
                    "provider split paths must exist inside the declared dataset root"
                ) from error
            if not resolved_path.is_dir():
                raise EvaluationRuntimeError(
                    "provider split paths must be directories inside the declared dataset root"
                )
            if resolved_path.name != "images":
                raise EvaluationRuntimeError(
                    "provider split paths must identify an images directory inside the dataset root"
                )
            labels_path = resolved_path.parent / "labels"
            cache_path = resolved_path.parent / "labels.cache"
            try:
                labels_path.relative_to(root)
                cache_path.relative_to(root)
            except ValueError as error:
                raise EvaluationRuntimeError(
                    "provider label and cache paths must remain inside the declared dataset root"
                ) from error
            _require_non_symlink_path(labels_path, "provider labels path")
            if not labels_path.is_dir():
                raise EvaluationRuntimeError(
                    "provider split images must have a sibling labels directory "
                    "inside the dataset root"
                )
    return root


@dataclass(frozen=True, slots=True)
class _CacheFingerprint:
    size: int
    sha256: str


def _discover_cache_paths(dataset_root: Path) -> frozenset[Path]:
    directories = [dataset_root]
    visited_directories = 0
    cache_paths: set[Path] = set()
    try:
        while directories:
            directory = directories.pop()
            visited_directories += 1
            if visited_directories > _MAX_CACHE_DIRECTORIES:
                raise EvaluationRuntimeError("provider dataset tree exceeds cache scan bounds")
            for entry in directory.iterdir():
                lexical_path = entry.absolute()
                if _is_link_or_junction(entry):
                    if entry.suffix == ".cache":
                        cache_paths.add(lexical_path)
                        continue
                    raise EvaluationRuntimeError("provider dataset tree must not contain symlinks")
                if entry.suffix == ".cache":
                    cache_paths.add(lexical_path)
                elif entry.is_dir():
                    directories.append(entry)
                if len(cache_paths) > _MAX_CACHE_FILES:
                    raise EvaluationRuntimeError("provider dataset has too many cache files")
    except EvaluationRuntimeError:
        raise
    except OSError as error:
        raise EvaluationRuntimeError("provider dataset cache state cannot be inspected") from error
    return frozenset(cache_paths)


def _cache_fingerprint(path: Path) -> _CacheFingerprint:
    try:
        if _is_link_or_junction(path) or not path.is_file():
            raise EvaluationRuntimeError("pre-existing provider cache must be a regular file")
        before = path.stat()
        if before.st_size > _MAX_CACHE_SNAPSHOT_BYTES:
            raise EvaluationRuntimeError("pre-existing provider cache exceeds snapshot bounds")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except EvaluationRuntimeError:
        raise
    except OSError as error:
        raise EvaluationRuntimeError("provider cache cannot be fingerprinted safely") from error
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EvaluationRuntimeError("provider cache changed while it was fingerprinted")
    return _CacheFingerprint(size=after.st_size, sha256=digest.hexdigest())


def _snapshot_existing_caches(dataset_root: Path) -> Mapping[Path, _CacheFingerprint]:
    paths = _discover_cache_paths(dataset_root)
    total_size = 0
    snapshot: dict[Path, _CacheFingerprint] = {}
    for path in sorted(paths):
        fingerprint = _cache_fingerprint(path)
        total_size += fingerprint.size
        if total_size > _MAX_CACHE_SNAPSHOT_BYTES:
            raise EvaluationRuntimeError("pre-existing provider caches exceed snapshot bounds")
        snapshot[path] = fingerprint
    return snapshot


CacheUnlink = Callable[[Path], None]


def _unlink_cache(path: Path) -> None:
    path.unlink()


def _cleanup_validation_caches(
    dataset_root: Path,
    snapshot: Mapping[Path, _CacheFingerprint],
    unlink_cache: CacheUnlink,
) -> None:
    try:
        current_paths = _discover_cache_paths(dataset_root)
        for path in sorted(current_paths.difference(snapshot)):
            unlink_cache(path)
        final_paths = _discover_cache_paths(dataset_root)
        if final_paths != frozenset(snapshot):
            raise EvaluationRuntimeError("provider cache cleanup did not restore the initial set")
        for path, expected in snapshot.items():
            if _cache_fingerprint(path) != expected:
                raise EvaluationRuntimeError("pre-existing provider cache integrity changed")
    except EvaluationRuntimeError:
        raise
    except OSError as error:
        raise EvaluationRuntimeError("generated provider cache could not be removed") from error


class ValidationWorkspace(Protocol):
    name: str

    def cleanup(self) -> None:
        """Remove the isolated provider output directory."""


ValidationWorkspaceFactory = Callable[[], ValidationWorkspace]


def _temporary_validation_workspace() -> ValidationWorkspace:
    return tempfile.TemporaryDirectory(prefix="smartsite-ai-validation-")


class UltralyticsValidationProvider:
    """Pinned, local-only Ultralytics implementation of ValidationProvider."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        provider_version_factory: ProviderVersionFactory | None = None,
        workspace_factory: ValidationWorkspaceFactory | None = None,
        cache_unlink: CacheUnlink | None = None,
    ) -> None:
        self._model_factory = model_factory or _ultralytics_model
        self._provider_version_factory = provider_version_factory or _ultralytics_version
        self._workspace_factory = workspace_factory or _temporary_validation_workspace
        self._cache_unlink = cache_unlink or _unlink_cache

    def validate(self, arguments: ProviderValidationArguments) -> object:
        provider_version = self._provider_version_factory()
        if provider_version != PINNED_ULTRALYTICS_VERSION:
            raise EvaluationRuntimeError(
                f"Ultralytics runtime must equal pinned version {PINNED_ULTRALYTICS_VERSION}"
            )
        _require_regular_local_file(arguments.model_path, "model_path")
        _require_regular_local_file(arguments.data_config_path, "data_config_path")
        document = _read_provider_data_config(arguments.data_config_path)
        dataset_root = _provider_dataset_root(arguments.data_config_path, document, arguments.split)
        cache_snapshot = _snapshot_existing_caches(dataset_root)
        workspace: ValidationWorkspace | None = None
        workspace_path: Path | None = None
        cleanup_errors: list[EvaluationRuntimeError] = []
        try:
            try:
                workspace = self._workspace_factory()
                workspace_path = Path(workspace.name)
            except OSError as error:
                raise EvaluationRuntimeError(
                    "isolated provider output workspace could not be created"
                ) from error
            if (
                not workspace_path.is_absolute()
                or _is_link_or_junction(workspace_path)
                or not workspace_path.is_dir()
                or any(workspace_path.iterdir())
            ):
                raise EvaluationRuntimeError(
                    "isolated provider output workspace must be a new empty local directory"
                )

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
                project=str(workspace_path),
                name="validation",
                exist_ok=False,
                save=False,
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
        finally:
            try:
                _cleanup_validation_caches(dataset_root, cache_snapshot, self._cache_unlink)
            except EvaluationRuntimeError as error:
                cleanup_errors.append(error)
            if workspace is not None:
                try:
                    workspace.cleanup()
                    if workspace_path is not None and workspace_path.exists():
                        raise OSError("workspace still exists after cleanup")
                except OSError:
                    cleanup_errors.append(
                        EvaluationRuntimeError(
                            "isolated provider output workspace could not be cleaned"
                        )
                    )
            if cleanup_errors:
                raise EvaluationRuntimeError(
                    "provider validation cleanup failed; validation result was discarded"
                ) from cleanup_errors[0]


__all__ = [
    "EvaluationRuntimeError",
    "OpenCvEvaluationFrameReader",
    "OpenCvEvaluationMedia",
    "OpenCvEvidenceExporter",
    "OpenCvOverlayRenderer",
    "UltralyticsValidationProvider",
    "ultralytics_runtime_sandbox",
]
