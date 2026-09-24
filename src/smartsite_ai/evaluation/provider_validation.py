"""Provider-neutral boundary for reproducible detector validation metrics.

This module deliberately contains no computer-vision provider imports.  A real
Ultralytics implementation is supplied at the application boundary; tests and
other callers may supply any object implementing :class:`ValidationProvider`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP

PINNED_ULTRALYTICS_VERSION = "8.4.155"

_BoundedDimension = Annotated[int, Field(ge=1, le=16_384)]
_CanonicalClassMap = tuple[tuple[int, str], ...]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class ProviderValidationArguments(_StrictFrozenModel):
    """Complete, immutable arguments used for one provider validation run."""

    model_path: Path
    data_config_path: Path
    task: Literal["detect"] = "detect"
    mode: Literal["val"] = "val"
    split: Literal["train", "val", "test"] = "test"
    image_size: tuple[_BoundedDimension, ...] = Field(min_length=2, max_length=2)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(gt=0.0, le=1.0)
    max_detections: int = Field(ge=1, le=65_536)
    batch_size: int = Field(ge=1, le=4_096)
    workers: int = Field(ge=0, le=256)
    device: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 1)
    deterministic: Literal[True] = True
    plots: Literal[False] = False
    save_json: Literal[False] = False

    @field_validator("model_path", "data_config_path")
    @classmethod
    def require_absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("path must be absolute")
        return value

    @field_validator("device")
    @classmethod
    def require_non_blank_device(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("device must be non-blank")
        return value


class ProviderValidationMetrics(_StrictFrozenModel):
    """Validated provider output retained in the reproducible evaluation report."""

    provider_name: str = Field(min_length=1, max_length=64)
    provider_version: str = Field(min_length=1, max_length=64)
    class_map: _CanonicalClassMap = Field(min_length=5, max_length=5)
    ap50: float = Field(ge=0.0, le=1.0)
    ap50_95: float = Field(ge=0.0, le=1.0)

    @field_validator("provider_name")
    @classmethod
    def require_ultralytics(cls, value: str) -> str:
        if value != "ultralytics":
            raise ValueError("provider_name must be ultralytics")
        return value

    @field_validator("provider_version")
    @classmethod
    def require_pinned_version(cls, value: str) -> str:
        if value != PINNED_ULTRALYTICS_VERSION:
            raise ValueError(
                f"provider_version must equal pinned version {PINNED_ULTRALYTICS_VERSION}"
            )
        return value

    @field_validator("class_map")
    @classmethod
    def require_canonical_class_map(cls, value: _CanonicalClassMap) -> _CanonicalClassMap:
        if value != CANONICAL_PPE_CLASS_MAP:
            raise ValueError("class_map must equal the canonical PPE class set and order")
        return value

    @model_validator(mode="after")
    def validate_metric_order(self) -> Self:
        if self.ap50_95 > self.ap50:
            raise ValueError("ap50_95 must not exceed ap50")
        return self


class ProviderValidationReport(_StrictFrozenModel):
    """Arguments and metrics from the same provider invocation."""

    arguments: ProviderValidationArguments
    metrics: ProviderValidationMetrics


class ProviderValidationError(RuntimeError):
    """Safe public failure raised at the provider boundary."""


class ValidationProvider(Protocol):
    def validate(self, arguments: ProviderValidationArguments) -> object:
        """Validate one model and return provider facts and AP metrics."""
        ...


class ProviderValidationAdapter:
    """Validate provider output without importing or trusting provider internals."""

    def __init__(self, provider: ValidationProvider) -> None:
        self._provider = provider

    def validate(self, arguments: ProviderValidationArguments) -> ProviderValidationReport:
        try:
            result = self._provider.validate(arguments)
        except Exception as error:
            raise ProviderValidationError("provider validation failed") from error

        try:
            metrics = ProviderValidationMetrics.model_validate(result)
        except ValidationError as error:
            raise ProviderValidationError(f"invalid provider validation result: {error}") from error

        return ProviderValidationReport(arguments=arguments, metrics=metrics)
