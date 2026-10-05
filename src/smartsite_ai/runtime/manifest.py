"""Strict manifest for a bounded multi-camera MF05/MF06 runtime."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from smartsite_ai.inference.ppe_profiles import ExperimentalPpeProfile
from smartsite_ai.ingestion.config import (
    _DEVICE_INDEX_RE,
    StreamConfig,
    _is_absolute_video_path,
)
from smartsite_ai.training.dataset_integrity import is_link_like

_ENV_NAME_RE = re.compile(r"^SMARTSITE_AI_[A-Z0-9_]{1,80}$")
_STREAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_MESSAGES = frozenset(
    {
        "manifest schema version is unsupported",
        "manifest must contain 1 to 3 cameras",
        "duplicate stream id",
        "duplicate camera id",
        "duplicate camera external id",
        "duplicate outbox path",
        "duplicate evidence directory",
        "duplicate camera source",
        "duplicate camera source reference",
        "camera source credentials belong in an environment reference",
        "camera source environment name is invalid",
        "camera runtime manifest is invalid",
    }
)


class CameraRuntimeError(RuntimeError):
    """Operator-facing manifest or runtime failure whose text is safe to print."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )


class PathCameraSource(_StrictModel):
    kind: Literal["path"]
    path: str = Field(min_length=1, max_length=1024)


class EnvCameraSource(_StrictModel):
    kind: Literal["env"]
    name: str = Field(min_length=1, max_length=96)


CameraSource = Annotated[PathCameraSource | EnvCameraSource, Field(discriminator="kind")]


class CameraManifestEntry(_StrictModel):
    stream_id: str = Field(alias="streamId", pattern=_STREAM_ID_RE.pattern)
    camera_id: UUID = Field(alias="cameraId")
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    ppe_region_id: UUID = Field(alias="ppeRegionId")
    source: CameraSource
    live: bool = False
    target_fps: float | None = Field(default=None, alias="targetFps", gt=0, le=120)
    outbox: str = Field(min_length=1, max_length=1024)
    evidence_dir: str | None = Field(
        default=None, alias="evidenceDir", min_length=1, max_length=1024
    )

    @model_validator(mode="after")
    def require_absolute_runtime_paths(self) -> CameraManifestEntry:
        if not _is_portable_absolute_path(self.outbox):
            raise ValueError("camera runtime manifest is invalid")
        if self.evidence_dir is not None and not _is_portable_absolute_path(self.evidence_dir):
            raise ValueError("camera runtime manifest is invalid")
        return self


class CameraRuntimeManifest(_StrictModel):
    schema_version: Literal["1"] = Field(alias="schemaVersion")
    model_spec: str = Field(alias="modelSpec", min_length=1, max_length=1024)
    experimental_model_profile: ExperimentalPpeProfile | None = Field(
        default=None, alias="experimentalModelProfile"
    )
    observation_schema_version: Literal["1.0.0", "1.1.0"] = Field(
        default="1.0.0", alias="observationSchemaVersion"
    )
    poll_interval_seconds: float = Field(default=5.0, alias="pollIntervalSeconds", gt=0, le=3600)
    stale_after_seconds: float = Field(default=60.0, alias="staleAfterSeconds", gt=0, le=86_400)
    delivery_interval_seconds: float = Field(
        default=0.25, alias="deliveryIntervalSeconds", gt=0, le=60
    )
    evidence_max_bytes: int = Field(
        default=1_048_576, alias="evidenceMaxBytes", gt=0, le=20_000_000
    )
    cameras: tuple[CameraManifestEntry, ...]

    @model_validator(mode="after")
    def validate_bounds(self) -> CameraRuntimeManifest:
        if (
            self.experimental_model_profile is not None
            and self.observation_schema_version != "1.1.0"
        ):
            raise ValueError("camera runtime manifest is invalid")
        if not 1 <= len(self.cameras) <= 3:
            raise ValueError("manifest must contain 1 to 3 cameras")
        if self.stale_after_seconds < self.poll_interval_seconds:
            raise ValueError("camera runtime manifest is invalid")
        _reject_duplicates(self.cameras)
        if not _is_portable_absolute_path(self.model_spec):
            raise ValueError("camera runtime manifest is invalid")
        return self


class BoundCamera:
    """One camera whose source has been resolved and checked without being logged."""

    def __init__(
        self,
        entry: CameraManifestEntry,
        *,
        source: str,
        source_config: StreamConfig,
        outbox: Path,
        evidence_dir: Path | None,
    ) -> None:
        self.entry = entry
        self.source = source
        self.source_config = source_config
        self.outbox = outbox
        self.evidence_dir = evidence_dir

    @property
    def stream_id(self) -> str:
        return self.entry.stream_id

    @property
    def camera_external_id(self) -> str:
        return self.entry.camera_external_id


class BoundManifest:
    def __init__(
        self,
        manifest: CameraRuntimeManifest,
        cameras: tuple[BoundCamera, ...],
        *,
        model_spec: Path,
    ) -> None:
        self.manifest = manifest
        self.cameras = cameras
        self.model_spec = model_spec


def parse_manifest(document: object) -> CameraRuntimeManifest:
    """Validate a manifest document. Paths and secrets are resolved separately."""

    if isinstance(document, dict) and document.get("schemaVersion") not in (None, "1"):
        raise CameraRuntimeError("manifest schema version is unsupported")
    try:
        return CameraRuntimeManifest.model_validate(document)
    except ValidationError as exc:
        raise CameraRuntimeError(_safe_validation_message(exc)) from None


def load_manifest_bytes(payload: bytes) -> CameraRuntimeManifest:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CameraRuntimeError("camera runtime manifest is invalid") from exc
    return parse_manifest(document)


def bind_manifest(
    manifest: CameraRuntimeManifest,
    environ: Mapping[str, str],
) -> BoundManifest:
    """Resolve env-backed sources and require isolated absolute runtime paths."""

    model_spec = _existing_file(Path(manifest.model_spec), "model spec")
    bound: list[BoundCamera] = []
    seen_sources: set[str] = set()
    for entry in manifest.cameras:
        source = _resolve_source(entry, environ)
        identity = _resolved_source_identity(source)
        if identity in seen_sources:
            raise CameraRuntimeError("duplicate camera source")
        seen_sources.add(identity)
        try:
            source_config = StreamConfig(
                stream_id=entry.stream_id,
                camera_external_id=entry.camera_external_id,
                source_url=source,
                target_fps=entry.target_fps,
                is_live=entry.live,
                pace_replay=not entry.live,
                max_consecutive_failures=None if entry.live else 1,
            )
        except ValueError:
            raise CameraRuntimeError(f"camera {entry.stream_id} source is invalid") from None
        outbox = _outbox_path(Path(entry.outbox))
        evidence_dir = None
        if entry.evidence_dir is not None:
            evidence_dir = _existing_dir(Path(entry.evidence_dir), "evidence directory")
        bound.append(
            BoundCamera(
                entry,
                source=source,
                source_config=source_config,
                outbox=outbox,
                evidence_dir=evidence_dir,
            )
        )
    return BoundManifest(manifest, tuple(bound), model_spec=model_spec)


def _reject_duplicates(cameras: tuple[CameraManifestEntry, ...]) -> None:
    stream_ids: set[str] = set()
    camera_ids: set[str] = set()
    external_ids: set[str] = set()
    outboxes: set[str] = set()
    evidence_dirs: set[str] = set()
    source_keys: set[str] = set()
    for camera in cameras:
        _add_unique(stream_ids, camera.stream_id.casefold(), "duplicate stream id")
        _add_unique(camera_ids, str(camera.camera_id).casefold(), "duplicate camera id")
        _add_unique(
            external_ids,
            camera.camera_external_id.casefold(),
            "duplicate camera external id",
        )
        _add_unique(outboxes, _path_key(camera.outbox), "duplicate outbox path")
        if camera.evidence_dir is not None:
            _add_unique(
                evidence_dirs,
                _path_key(camera.evidence_dir),
                "duplicate evidence directory",
            )
        source_key, source_error = _source_key(camera.source)
        if source_key in source_keys:
            raise ValueError(source_error)
        source_keys.add(source_key)


def _source_key(source: CameraSource) -> tuple[str, str]:
    if isinstance(source, EnvCameraSource):
        if _ENV_NAME_RE.fullmatch(source.name) is None:
            raise ValueError("camera source environment name is invalid")
        return f"env:{source.name}", "duplicate camera source reference"
    if not _manifest_path_allowed(source.path):
        raise ValueError("camera source credentials belong in an environment reference")
    return f"path:{_path_key(source.path)}", "duplicate camera source"


def _manifest_path_allowed(value: str) -> bool:
    if "://" in value or value.lower().startswith("file:"):
        return False
    stripped = value.strip()
    return bool(_DEVICE_INDEX_RE.fullmatch(stripped) or _is_absolute_video_path(stripped))


def _resolved_source_identity(source: str) -> str:
    if _manifest_path_allowed(source):
        return f"path:{_path_key(source)}"
    return f"secret:{source}"


def _resolve_source(entry: CameraManifestEntry, environ: Mapping[str, str]) -> str:
    source = entry.source
    if isinstance(source, EnvCameraSource):
        value = environ.get(source.name)
        if value is None or not value.strip():
            raise CameraRuntimeError(
                f"camera {entry.stream_id} source environment variable {source.name} is unset"
            )
        return value
    return source.path


def _add_unique(seen: set[str], key: str, message: str) -> None:
    if key in seen:
        raise ValueError(message)
    seen.add(key)


def _path_key(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve(strict=False)))


def _is_portable_absolute_path(value: str) -> bool:
    """Recognize explicit POSIX and Windows absolute paths on either host OS."""

    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _existing_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CameraRuntimeError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise CameraRuntimeError(f"{label} path must identify a regular non-symlink file")
    if is_link_like(path):
        raise CameraRuntimeError(f"{label} path must identify a regular non-symlink file")
    return path


def _existing_dir(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise CameraRuntimeError(f"{label} path must be absolute")
    if not path.is_dir() or is_link_like(path):
        raise CameraRuntimeError(
            f"{label} path must identify an existing regular non-symlink directory"
        )
    return path


def _outbox_path(path: Path) -> Path:
    if not path.is_absolute():
        raise CameraRuntimeError("outbox path must be absolute")
    if not path.parent.is_dir() or is_link_like(path.parent):
        raise CameraRuntimeError("outbox parent directory must already exist")
    if path.is_symlink() or (path.exists() and (not path.is_file() or is_link_like(path))):
        raise CameraRuntimeError("outbox path must identify a regular non-symlink file")
    return path


def _safe_validation_message(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors():
        message = str(error.get("msg", ""))
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        if message not in _SAFE_MESSAGES:
            return "camera runtime manifest is invalid"
        if message not in messages:
            messages.append(message)
    if not messages:
        return "camera runtime manifest is invalid"
    return "; ".join(messages)


__all__ = [
    "BoundCamera",
    "BoundManifest",
    "CameraRuntimeError",
    "CameraRuntimeManifest",
    "bind_manifest",
    "load_manifest_bytes",
    "parse_manifest",
]
