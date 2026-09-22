"""Local model artifact metadata and integrity verification."""

import hashlib
import hmac
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NO_NUL_PATTERN = r"^[^\x00]+$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_CLASS_ID = 2_147_483_647
_MAX_IMAGE_SIZE = 16_384
_HASH_CHUNK_SIZE = 1024 * 1024

ClassMapEntry = tuple[
    Annotated[int, Field(ge=0, le=_MAX_CLASS_ID)],
    Annotated[str, Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)],
]
ImageSize = tuple[
    Annotated[int, Field(ge=1, le=_MAX_IMAGE_SIZE)],
    Annotated[int, Field(ge=1, le=_MAX_IMAGE_SIZE)],
]


class ArtifactValidationError(ValueError):
    """The declared model artifact cannot be safely used."""


class ArtifactNotFoundError(ArtifactValidationError):
    """The declared local model artifact does not exist."""


class ArtifactChecksumMismatchError(ArtifactValidationError):
    """The local model artifact does not match its declared checksum."""

    def __init__(self, expected_sha256: str, actual_sha256: str) -> None:
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        super().__init__(
            "model artifact checksum does not match declared SHA-256: "
            f"expected {expected_sha256}, actual {actual_sha256}"
        )


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class ModelArtifactSpec(_StrictFrozenModel):
    """Declared metadata for one local detector artifact."""

    artifact_id: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    version: str = Field(min_length=1, max_length=64, pattern=_NO_NUL_PATTERN)
    model_family: str = Field(min_length=1, max_length=64, pattern=_NO_NUL_PATTERN)
    artifact_path: Path
    sha256: str = Field(pattern=_SHA256_PATTERN)
    source_url: str = Field(min_length=1, max_length=2048, pattern=_NO_NUL_PATTERN)
    license: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)
    class_map: tuple[ClassMapEntry, ...] = Field(min_length=1, max_length=1024)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    image_size: ImageSize
    device: str = Field(min_length=1, max_length=128, pattern=_NO_NUL_PATTERN)

    @field_validator("artifact_path")
    @classmethod
    def validate_absolute_artifact_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("artifact_path must be absolute")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_https_source_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source_url must be an HTTPS URL")
        return value

    @field_validator("class_map")
    @classmethod
    def validate_unique_class_ids(
        cls, value: tuple[ClassMapEntry, ...]
    ) -> tuple[ClassMapEntry, ...]:
        class_ids = tuple(class_id for class_id, _ in value)
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("class_map class IDs must be unique")
        return value


class VerifiedModelArtifact(ModelArtifactSpec):
    """A model artifact whose local bytes match its declared SHA-256 digest."""

    resolved_path: Path
    actual_sha256: str = Field(pattern=_SHA256_PATTERN)


def verify_model_artifact(spec: ModelArtifactSpec) -> VerifiedModelArtifact:
    """Resolve and verify a configured local artifact without loading a model runtime."""
    artifact_path = spec.artifact_path
    if not artifact_path.is_absolute():
        raise ArtifactValidationError("model artifact path must be absolute")
    if artifact_path.is_symlink():
        raise ArtifactValidationError("model artifact path must not be a symlink")
    if not artifact_path.exists():
        raise ArtifactNotFoundError("model artifact path does not exist")
    if not artifact_path.is_file():
        raise ArtifactValidationError("model artifact path must be a regular file")

    resolved_path = artifact_path.resolve(strict=True)
    actual_sha256 = _sha256_file(resolved_path)
    if not hmac.compare_digest(spec.sha256, actual_sha256):
        raise ArtifactChecksumMismatchError(spec.sha256, actual_sha256)

    return VerifiedModelArtifact.model_validate(
        {
            **spec.model_dump(),
            "resolved_path": resolved_path,
            "actual_sha256": actual_sha256,
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact_file:
            while chunk := artifact_file.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ArtifactNotFoundError("model artifact path does not exist") from error
    except OSError as error:
        raise ArtifactValidationError("model artifact path cannot be read") from error
    return digest.hexdigest()
