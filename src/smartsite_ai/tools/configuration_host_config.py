"""Bounded private operator manifest; source and token are environment references."""

import json
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from smartsite_ai.ingestion.config import StreamConfig

_ENV = r"^SMARTSITE_AI_[A-Z0-9_]{1,80}$"
_CODES = frozenset(
    {
        "CONFIGURATION_HOST_INVALID_CONFIG",
        "CONFIGURATION_HOST_BIND_UNAVAILABLE",
        "CONFIGURATION_HOST_SERVER_FAILED",
        "CONFIGURATION_HOST_CAPTURE_FAILED",
        "CONFIGURATION_HOST_STOP_UNRESOLVED",
        "CONFIGURATION_HOST_INTERRUPTED",
    }
)


class ConfigurationHostError(RuntimeError):
    """Allowlisted code only, never native exception/configuration contents."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _CODES else "CONFIGURATION_HOST_SERVER_FAILED"
        super().__init__(self.code)


class _Manifest(BaseModel):
    model_config = ConfigDict(
        strict=True, extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    stream_id: str = Field(alias="streamId", min_length=1, max_length=128)
    camera_external_id: str = Field(alias="cameraExternalId", min_length=1, max_length=128)
    source_kind: Literal["LIVE", "REPLAY"] = Field(alias="sourceKind")
    source_env: str = Field(alias="sourceEnv", pattern=_ENV)
    token_env: str = Field(alias="tokenEnv", pattern=_ENV)
    resource_id: UUID = Field(alias="resourceId")
    lock_directory: Path = Field(alias="lockDirectory")
    port: int = Field(default=0, ge=0, le=65535)
    duration_seconds: float = Field(alias="durationSeconds", gt=0, le=86400)

    @field_validator("stream_id", "camera_external_id")
    @classmethod
    def printable_identity(cls, value: str) -> str:
        if not value.strip() or any(unicodedata.category(c) in {"Cc", "Cs"} for c in value):
            raise ValueError("Invalid source identity")
        return value


@dataclass(frozen=True, repr=False)
class HostConfiguration:
    stream: StreamConfig
    source_kind: Literal["LIVE", "REPLAY"]
    resource_id: UUID
    lock_directory: Path
    token: SecretStr
    port: int
    duration_seconds: float


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate manifest key")
        value[key] = item
    return value


def load_host_configuration(
    path: Path, environ: Mapping[str, str], *, controlled_stop: bool = False
) -> HostConfiguration:
    """Validate before any socket, capture/source factory or server is constructed."""
    try:
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Regular configuration file required")
        with path.open("rb") as stream:
            data = stream.read(16385)
        if len(data) > 16384:
            raise ValueError("Configuration exceeds byte budget")
        json.loads(data, object_pairs_hook=_unique_object)
        manifest = _Manifest.model_validate_json(data)
        if manifest.duration_seconds > 3600 and controlled_stop is not True:
            raise ValueError("Long sessions require an explicit stop control")
        raw, token = environ[manifest.source_env], environ[manifest.token_env]
        if (
            manifest.source_env == manifest.token_env
            or not 32 <= len(token) <= 4096
            or not token.isascii()
            or any(c.isspace() for c in token)
        ):
            raise ValueError("Invalid server-only environment binding")
        directory = manifest.lock_directory.resolve(strict=True)
        if (
            not manifest.lock_directory.is_absolute()
            or not directory.is_dir()
            or str(directory).startswith("\\\\")
        ):
            raise ValueError("Existing local private lock directory required")
        source = raw.strip()
        local_video = Path(source).is_absolute() and ".." not in source
        if manifest.source_kind == "REPLAY":
            if not local_video or not Path(source).is_file() or source.startswith("\\\\"):
                raise ValueError("REPLAY requires a local video file")
        elif local_video and source.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            raise ValueError("Local video cannot be declared LIVE")
        stream = StreamConfig(
            stream_id=manifest.stream_id,
            camera_external_id=manifest.camera_external_id,
            source_url=SecretStr(raw),
            is_live=manifest.source_kind == "LIVE",
            pace_replay=manifest.source_kind == "REPLAY",
            max_queue_size=1,
        )
        return HostConfiguration(
            stream,
            manifest.source_kind,
            manifest.resource_id,
            directory,
            SecretStr(token),
            manifest.port,
            manifest.duration_seconds,
        )
    except (OSError, ValueError, KeyError, TypeError):
        raise ConfigurationHostError("CONFIGURATION_HOST_INVALID_CONFIG") from None
