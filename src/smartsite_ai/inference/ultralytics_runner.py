"""Explicit-lifecycle Ultralytics adapter for packed BGR24 frames."""

import os
from collections.abc import Callable, Mapping
from importlib.metadata import version as distribution_version
from math import isfinite
from pathlib import Path

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    InferenceResultError,
    RawYoloDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope

_MAX_RAW_DETECTIONS = 1_024
_RAW_RESULT_LIMIT = _MAX_RAW_DETECTIONS + 1

ModelFactory = Callable[[Path], object]
ImageFactory = Callable[[FrameEnvelope], object]
ProviderVersionFactory = Callable[[], str]


class UltralyticsYoloRunner:
    """Run a verified local Ultralytics model only after explicit loading."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        image_factory: ImageFactory | None = None,
        provider_version_factory: ProviderVersionFactory | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._image_factory = image_factory or _bgr_image
        self._provider_version_factory = provider_version_factory or _ultralytics_version
        self._model: object | None = None
        self._artifact: VerifiedModelArtifact | None = None

    @property
    def provider_metadata(self) -> dict[str, object]:
        """Return facts read from the loaded provider checkpoint."""

        model = self._model
        if model is None:
            raise DetectorUnavailableError("Ultralytics runner is not loaded")
        return _extract_provider_metadata(model, self._provider_version_factory())

    def load(self, artifact: VerifiedModelArtifact) -> None:
        """Construct the model from an already verified local artifact."""
        if self._model is not None:
            raise DetectorUnavailableError("Ultralytics runner is already loaded")

        try:
            model = self._model_factory(artifact.resolved_path)
        except DetectorUnavailableError:
            raise
        except Exception as error:
            raise DetectorUnavailableError("Ultralytics model construction failed") from error

        self._model = model
        self._artifact = artifact

    def predict(self, frame: FrameEnvelope) -> tuple[RawYoloDetection, ...]:
        """Predict provider-neutral source-frame rows for one BGR24 frame."""
        model = self._model
        artifact = self._artifact
        if model is None or artifact is None:
            raise DetectorUnavailableError("Ultralytics runner is not loaded")
        if frame.pixel_format != "BGR24":
            raise DetectorUnavailableError("Ultralytics runner requires BGR24 frames")

        image = self._image_factory(frame)
        try:
            results = model.predict(
                source=image,
                conf=artifact.confidence_threshold,
                iou=artifact.iou_threshold,
                imgsz=_provider_image_size(artifact.image_size),
                device=artifact.device,
                verbose=False,
            )
        except Exception as error:
            raise DetectorUnavailableError("Ultralytics provider prediction failed") from error

        try:
            return _translate_first_result(results, artifact)
        except InferenceResultError:
            raise
        except Exception as error:
            raise DetectorUnavailableError(
                "Ultralytics provider returned an invalid result"
            ) from error

    def close(self) -> None:
        """Drop references to the model and its verified artifact."""
        self._model = None
        self._artifact = None


def _provider_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Convert a declared `(width, height)` pair into Ultralytics `[height, width]`."""

    width, height = image_size
    return (height, width)


def _ultralytics_version() -> str:
    return distribution_version("ultralytics")


def _extract_provider_metadata(model: object, provider_version: str) -> dict[str, object]:
    task = getattr(model, "task", None)
    names = getattr(model, "names", None)
    network = getattr(model, "model", None)
    yaml = getattr(network, "yaml", None)
    if not isinstance(yaml, Mapping):
        raise DetectorUnavailableError("Ultralytics checkpoint metadata is unavailable")

    yaml_file = yaml.get("yaml_file")
    yaml_stem = Path(yaml_file).stem if isinstance(yaml_file, str) else ""
    if yaml_stem not in {"yolo11", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}:
        raise DetectorUnavailableError(
            "Ultralytics checkpoint does not prove a YOLO11 architecture"
        )
    scale = yaml.get("scale")
    if scale not in {"n", "s", "m", "l", "x"}:
        raise DetectorUnavailableError("Ultralytics checkpoint has an invalid YOLO11 variant")
    if yaml_stem != "yolo11" and yaml_stem != f"yolo11{scale}":
        raise DetectorUnavailableError(
            "Ultralytics checkpoint YOLO11 variant metadata is inconsistent"
        )
    if not isinstance(task, str):
        raise DetectorUnavailableError("Ultralytics checkpoint task metadata is unavailable")
    if not isinstance(names, Mapping):
        raise DetectorUnavailableError("Ultralytics checkpoint class metadata is unavailable")

    class_map: dict[str, str] = {}
    for class_id, class_name in names.items():
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise DetectorUnavailableError("Ultralytics checkpoint class IDs must be integers")
        if not isinstance(class_name, str):
            raise DetectorUnavailableError("Ultralytics checkpoint class names must be strings")
        class_map[str(class_id)] = class_name

    return {
        "providerName": "ultralytics",
        "providerVersion": provider_version,
        "architecture": "yolo11",
        "variant": scale,
        "task": task,
        "classMap": class_map,
    }


def _default_model_factory(path: Path) -> object:
    """Load one verified checkpoint without provider path rewrite, download, or install."""

    _require_exact_local_file(path)
    try:
        return _load_exact_ultralytics_model(path)
    except DetectorUnavailableError:
        raise
    except Exception as error:
        raise DetectorUnavailableError("Ultralytics model construction failed") from error


def _require_exact_local_file(path: Path) -> None:
    """Reject a verified path that is no longer the exact regular file that was checked."""

    if not path.is_absolute():
        raise DetectorUnavailableError("verified model artifact path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise DetectorUnavailableError("verified model artifact is unavailable")


def _load_exact_ultralytics_model(path: Path) -> object:
    """Construct YOLO from the exact local path with network and auto-install closed."""

    import ultralytics.nn.tasks as tasks
    import ultralytics.utils as ultralytics_utils
    import ultralytics.utils.checks as checks
    import ultralytics.utils.downloads as downloads
    from ultralytics import YOLO

    exact = Path(path)

    def exact_asset(file: str | Path, *_args: object, **_kwargs: object) -> str:
        if _path_key(file) != _path_key(exact):
            raise DetectorUnavailableError("provider requested a different model file")
        _require_exact_local_file(exact)
        return str(exact)

    original_requirements = checks.check_requirements

    def offline_requirements(*args: object, **kwargs: object) -> bool:
        install = bool(kwargs.get("install", True))
        satisfied = original_requirements(*args, **{**kwargs, "install": False})
        if install and satisfied is False:
            raise DetectorUnavailableError(
                "verified model requires a missing dependency and auto-install is disabled"
            )
        return bool(satisfied)

    previous_autoinstall = os.environ.get("YOLO_AUTOINSTALL")
    original_download = downloads.attempt_download_asset
    original_tasks_requirements = tasks.check_requirements
    original_utils_autoinstall = ultralytics_utils.AUTOINSTALL
    original_checks_autoinstall = checks.AUTOINSTALL
    os.environ["YOLO_AUTOINSTALL"] = "false"
    downloads.attempt_download_asset = exact_asset
    checks.check_requirements = offline_requirements
    tasks.check_requirements = offline_requirements
    ultralytics_utils.AUTOINSTALL = False
    checks.AUTOINSTALL = False
    try:
        _require_exact_local_file(exact)
        return YOLO(exact)
    finally:
        downloads.attempt_download_asset = original_download
        checks.check_requirements = original_requirements
        tasks.check_requirements = original_tasks_requirements
        ultralytics_utils.AUTOINSTALL = original_utils_autoinstall
        checks.AUTOINSTALL = original_checks_autoinstall
        if previous_autoinstall is None:
            os.environ.pop("YOLO_AUTOINSTALL", None)
        else:
            os.environ["YOLO_AUTOINSTALL"] = previous_autoinstall


def _path_key(path: str | Path) -> str:
    """Compare provider paths without resolving links or removing characters."""

    return os.path.normcase(os.path.normpath(str(path)))


def _bgr_image(frame: FrameEnvelope) -> object:
    """Make the read-only provider image view from immutable packed BGR bytes."""
    import numpy as np

    return np.frombuffer(frame.buffer, dtype=np.uint8).reshape((frame.height, frame.width, 3))


def _translate_first_result(
    results: object, artifact: VerifiedModelArtifact
) -> tuple[RawYoloDetection, ...]:
    first_result = next(iter(results), None)
    if first_result is None:
        return ()

    boxes = first_result.boxes
    if boxes is None:
        return ()
    names = first_result.names
    class_map = dict(artifact.class_map)
    rows: list[RawYoloDetection] = []
    for coordinates, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
    ):
        if len(rows) == _RAW_RESULT_LIMIT:
            break
        normalized_class_id = _class_id(class_id)
        _verify_class_name(names, normalized_class_id, class_map)
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise InferenceResultError("Ultralytics provider box must contain four coordinates")
        x1, y1, x2, y2 = (_finite_float(value, "coordinate") for value in coordinates)
        rows.append(
            RawYoloDetection(
                class_id=normalized_class_id,
                confidence=_finite_float(confidence, "confidence"),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return tuple(rows)


def _class_id(value: object) -> int:
    if isinstance(value, bool):
        raise InferenceResultError("Ultralytics provider class ID must be an integer")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceResultError("Ultralytics provider class ID must be an integer") from error
    if not isfinite(numeric_value) or not numeric_value.is_integer():
        raise InferenceResultError("Ultralytics provider class ID must be an integer")
    return int(numeric_value)


def _finite_float(value: object, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise InferenceResultError(f"Ultralytics provider {label} must be numeric") from error
    if not isfinite(numeric_value):
        raise InferenceResultError(f"Ultralytics provider {label} must be finite")
    return numeric_value


def _verify_class_name(names: object, class_id: int, class_map: dict[int, str]) -> None:
    if not isinstance(names, Mapping):
        raise InferenceResultError("Ultralytics provider class names must be a mapping")
    expected_name = class_map.get(class_id)
    provider_name = names.get(class_id)
    if expected_name is None or provider_name != expected_name:
        raise InferenceResultError(
            f"Ultralytics provider class name disagrees for class ID {class_id}"
        )
