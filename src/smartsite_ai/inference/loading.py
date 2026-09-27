"""Explicit, side-effect-free loading for verified YOLO11 detector artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    field_validator,
)

from smartsite_ai.inference.artifacts import (
    ArtifactNotFoundError,
    ArtifactValidationError,
    ModelArtifactSpec,
    VerifiedModelArtifact,
    verify_model_artifact,
)
from smartsite_ai.inference.yolo import RawYoloDetection, Yolo11Detector
from smartsite_ai.ingestion.envelope import FrameEnvelope

CANONICAL_PPE_CLASS_MAP: tuple[tuple[int, str], ...] = (
    (0, "Person"),
    (1, "Hardhat"),
    (2, "NO-Hardhat"),
    (3, "Safety Vest"),
    (4, "NO-Safety Vest"),
)

_CANONICAL_PPE_JSON_CLASS_MAP = {
    str(class_id): class_name for class_id, class_name in CANONICAL_PPE_CLASS_MAP
}
_MAX_SPEC_BYTES = 256 * 1024
_NO_NUL_PATTERN = r"^[^\x00]+$"
_MODEL_FAMILIES = frozenset({"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"})
_SECRET_FIELD_NAMES = frozenset(
    {"token", "password", "credential", "authorization", "apikey", "signedurl"}
)


def _class_map_items(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


ProviderClassMap = Annotated[
    tuple[
        tuple[
            Annotated[str, Field(min_length=1, max_length=16, pattern=r"^[0-9]+$")],
            Annotated[str, Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)],
        ],
        ...,
    ],
    BeforeValidator(_class_map_items),
    PlainSerializer(lambda value: dict(value), return_type=dict[str, str]),
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_by_alias=True,
        validate_by_name=False,
    )


class _ArtifactSpecDocument(_StrictFrozenModel):
    artifact_id: str = Field(
        alias="artifactId", min_length=1, max_length=128, pattern=_NO_NUL_PATTERN
    )
    version: str = Field(min_length=1, max_length=64, pattern=_NO_NUL_PATTERN)
    model_family: str = Field(
        alias="modelFamily", min_length=1, max_length=64, pattern=_NO_NUL_PATTERN
    )
    artifact_path: str = Field(
        alias="artifactPath", min_length=1, max_length=4096, pattern=_NO_NUL_PATTERN
    )
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str = Field(
        alias="sourceUrl", min_length=1, max_length=2048, pattern=_NO_NUL_PATTERN
    )
    license: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    class_map: dict[str, str] = Field(alias="classMap")
    confidence_threshold: float = Field(alias="confidenceThreshold", ge=0.0, le=1.0)
    iou_threshold: float = Field(alias="iouThreshold", ge=0.0, le=1.0)
    image_size: list[Annotated[int, Field(ge=1, le=16_384)]] = Field(
        alias="imageSize", min_length=2, max_length=2
    )
    device: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)

    @field_validator("model_family")
    @classmethod
    def validate_yolo11_family(cls, value: str) -> str:
        if value not in _MODEL_FAMILIES:
            raise ValueError("modelFamily must be an exact YOLO11 detector family")
        return value

    @field_validator("artifact_path")
    @classmethod
    def validate_absolute_local_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("artifactPath must be an absolute local path")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_public_https_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("sourceUrl must be a public HTTPS URL without credentials or query")
        return value

    @field_validator("class_map")
    @classmethod
    def validate_canonical_class_map(cls, value: dict[str, str]) -> dict[str, str]:
        if value != _CANONICAL_PPE_JSON_CLASS_MAP:
            raise ValueError("classMap must contain exactly the five canonical PPE classes")
        return value


class ProviderModelMetadata(_StrictFrozenModel):
    """Provider facts extracted from the loaded checkpoint, not copied from its spec."""

    provider_name: str = Field(
        alias="providerName", min_length=1, max_length=64, pattern=_NO_NUL_PATTERN
    )
    provider_version: str = Field(
        alias="providerVersion", min_length=1, max_length=64, pattern=_NO_NUL_PATTERN
    )
    architecture: str = Field(min_length=1, max_length=64, pattern=_NO_NUL_PATTERN)
    variant: Literal["n", "s", "m", "l", "x"]
    task: str = Field(min_length=1, max_length=32, pattern=_NO_NUL_PATTERN)
    class_map: ProviderClassMap = Field(alias="classMap", min_length=1, max_length=1024)

    @property
    def model_family(self) -> str:
        return f"{self.architecture}{self.variant}"


class LoadedYoloRunnerProtocol(Protocol):
    """Explicit lifecycle required by the shared loader."""

    @property
    def provider_metadata(self) -> object:
        """Return facts extracted from the provider after ``load``."""
        ...

    def load(self, artifact: VerifiedModelArtifact) -> None:
        """Load exactly the verified local artifact."""
        ...

    def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
        """Return provider-neutral detections."""
        ...

    def close(self) -> None:
        """Release provider resources."""
        ...


RunnerFactory = Callable[[], LoadedYoloRunnerProtocol]
LoadedDetectorStack = tuple[
    Yolo11Detector,
    LoadedYoloRunnerProtocol,
    VerifiedModelArtifact,
    Mapping[int, str],
]


def load_artifact_spec(path: Path) -> ModelArtifactSpec:
    """Load one bounded, strict camelCase JSON artifact contract without provider I/O."""

    raw = _read_bounded_json(path)
    _reject_secret_fields(raw)
    if not isinstance(raw, dict):
        raise ArtifactValidationError("artifact spec must be a JSON object")

    try:
        document = _ArtifactSpecDocument.model_validate(raw)
        return ModelArtifactSpec.model_validate(
            {
                "artifact_id": document.artifact_id,
                "version": document.version,
                "model_family": document.model_family,
                "artifact_path": Path(document.artifact_path),
                "sha256": document.sha256,
                "source_url": document.source_url,
                "license": document.license,
                "class_map": CANONICAL_PPE_CLASS_MAP,
                "confidence_threshold": document.confidence_threshold,
                "iou_threshold": document.iou_threshold,
                "image_size": tuple(document.image_size),
                "device": document.device,
            }
        )
    except ValidationError as error:
        raise ArtifactValidationError(f"invalid artifact spec: {error}") from error


def load_yolo11_detector(path: Path, *, runner_factory: RunnerFactory) -> LoadedDetectorStack:
    """Verify and load a YOLO11 detector through an injected explicit provider boundary."""

    spec = load_artifact_spec(path)
    return load_detector_artifact(
        spec,
        runner_factory=runner_factory,
        require_yolo11_metadata=True,
    )


def load_detector_artifact(
    spec: ModelArtifactSpec,
    *,
    runner_factory: RunnerFactory,
    require_yolo11_metadata: bool = False,
) -> LoadedDetectorStack:
    """Verify and load one local detector without changing its declared model identity.

    ``require_yolo11_metadata`` is reserved for the official evaluation gate. Local smoke
    commands may deliberately load a clearly labelled reference checkpoint without upgrading
    its claim to YOLO11.
    """

    artifact = verify_model_artifact(spec)
    runner = runner_factory()
    try:
        runner.load(artifact)
        if require_yolo11_metadata:
            metadata = _provider_metadata(runner.provider_metadata)
            _validate_provider_matches_spec(metadata, artifact)
        detector = Yolo11Detector(artifact, runner)
    except Exception:
        _close_safely(runner)
        raise

    class_map = MappingProxyType(dict(artifact.class_map))
    return detector, runner, artifact, class_map


def _provider_metadata(value: object) -> ProviderModelMetadata:
    if not isinstance(value, (dict, ProviderModelMetadata)):
        raise ArtifactValidationError("provider metadata is absent or invalid")
    try:
        return ProviderModelMetadata.model_validate(value)
    except ValidationError as error:
        raise ArtifactValidationError(f"provider metadata is invalid: {error}") from error


def _validate_provider_matches_spec(
    metadata: ProviderModelMetadata, artifact: VerifiedModelArtifact
) -> None:
    if metadata.architecture != "yolo11":
        raise ArtifactValidationError("provider metadata does not prove a YOLO11 architecture")
    if metadata.task != "detect":
        raise ArtifactValidationError("provider metadata does not prove the detection task")
    if metadata.model_family != artifact.model_family:
        raise ArtifactValidationError("provider metadata model family contradicts artifact spec")
    expected_class_map = {str(class_id): class_name for class_id, class_name in artifact.class_map}
    if dict(metadata.class_map) != expected_class_map:
        raise ArtifactValidationError("provider metadata class map contradicts artifact spec")


def _read_bounded_json(path: Path) -> Any:
    try:
        if path.is_symlink() or not path.is_file():
            if not path.exists():
                raise ArtifactNotFoundError("artifact spec path does not exist")
            raise ArtifactValidationError("artifact spec path must be a regular file")
        if path.stat().st_size > _MAX_SPEC_BYTES:
            raise ArtifactValidationError("artifact spec exceeds the 256 KiB limit")
        raw_text = path.read_text(encoding="utf-8")
    except (ArtifactValidationError, ArtifactNotFoundError):
        raise
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError("artifact spec cannot be read as UTF-8") from error

    try:
        return json.loads(raw_text, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ArtifactValidationError("artifact spec must contain valid JSON") from error


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_secret_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.replace("_", "").replace("-", "").lower()
            if any(normalized_key.endswith(name) for name in _SECRET_FIELD_NAMES):
                raise ArtifactValidationError("artifact spec contains a secret-bearing field")
            _reject_secret_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_secret_fields(child)


def _close_safely(runner: LoadedYoloRunnerProtocol) -> None:
    with suppress(Exception):
        runner.close()
