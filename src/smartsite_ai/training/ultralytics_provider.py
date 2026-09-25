"""Lazy Ultralytics provider for local official YOLO11s PPE fine-tuning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from smartsite_ai.evaluation.provider_validation import PINNED_ULTRALYTICS_VERSION
from smartsite_ai.inference.ultralytics_runner import (
    _extract_provider_metadata,
    _load_exact_ultralytics_model,
)
from smartsite_ai.tools.train_ppe import TrainingConfiguration, TrainingExecutionError


class TrainableModel(Protocol):
    def train(self, **arguments: object) -> object:
        """Run provider training."""


ModelFactory = Callable[[Path], TrainableModel]
VersionFactory = Callable[[], str]


def _provider_version() -> str:
    from importlib.metadata import version

    return version("ultralytics")


class UltralyticsTrainingProvider:
    """Load an exact local YOLO11s checkpoint and run deterministic training."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        version_factory: VersionFactory | None = None,
    ) -> None:
        self._model_factory = model_factory or _load_exact_ultralytics_model
        self._version_factory = version_factory or _provider_version

    def train(self, configuration: TrainingConfiguration) -> None:
        version = self._version_factory()
        if version != PINNED_ULTRALYTICS_VERSION:
            raise TrainingExecutionError(
                f"Ultralytics runtime must equal pinned version {PINNED_ULTRALYTICS_VERSION}"
            )
        model = self._model_factory(configuration.base_weights)
        metadata = _extract_provider_metadata(model, version)
        if not _is_official_yolo11s_detect(metadata):
            raise TrainingExecutionError(
                "base weights metadata must prove official Ultralytics YOLO11s detect architecture"
            )
        try:
            model.train(
                data=str(configuration.data_config),
                project=str(configuration.output_root),
                name=configuration.run_name,
                exist_ok=False,
                epochs=configuration.epochs,
                imgsz=configuration.image_size,
                batch=configuration.batch_size,
                patience=configuration.patience,
                seed=configuration.seed,
                device=configuration.device,
                deterministic=True,
                amp=False,
                workers=0,
                resume=False,
                plots=False,
                cache=False,
            )
        except TrainingExecutionError:
            raise
        except Exception as error:
            raise TrainingExecutionError("Ultralytics training failed") from error
        finally:
            _remove_generated_label_caches(configuration.data_config.parent)


def _is_official_yolo11s_detect(metadata: Mapping[str, object]) -> bool:
    return (
        metadata.get("providerName") == "ultralytics"
        and metadata.get("providerVersion") == PINNED_ULTRALYTICS_VERSION
        and metadata.get("architecture") == "yolo11"
        and metadata.get("variant") == "s"
        and metadata.get("task") == "detect"
    )


def _remove_generated_label_caches(dataset_root: Path) -> None:
    """Remove Ultralytics label indexes so the prepared dataset remains immutable."""

    try:
        for path in dataset_root.rglob("*.cache"):
            if path.is_symlink() or not path.is_file():
                continue
            path.unlink()
    except OSError as error:
        raise TrainingExecutionError(
            "could not remove generated Ultralytics dataset cache"
        ) from error
