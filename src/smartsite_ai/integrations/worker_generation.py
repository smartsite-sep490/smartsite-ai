"""Explicit operator-issued binding; no enrollment, model loading or camera startup."""

import json
import os
import re
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from smartsite_ai.domain.configuration_bundle import (
    _validate_lowercase_uuid,
    _validate_utc_millisecond_timestamp,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import FrameSource


class WorkerGenerationCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    camera_id: str = Field(alias="cameraId")
    worker_generation_id: str = Field(alias="workerGenerationId")
    stream_session_id: str = Field(alias="streamSessionId")
    secret: SecretStr
    expires_at: str = Field(alias="expiresAt")

    @field_validator("camera_id", "worker_generation_id", "stream_session_id")
    @classmethod
    def uuid(cls, value: str) -> str:
        return _validate_lowercase_uuid(value, field_name="generation identity")

    @field_validator("secret")
    @classmethod
    def secret_shape(cls, value: SecretStr) -> SecretStr:
        if re.fullmatch(r"[A-Za-z0-9_-]{43}", value.get_secret_value()) is None:
            raise ValueError("Invalid generation credential")
        return value

    @field_validator("expires_at")
    @classmethod
    def expiry(cls, value: str) -> str:
        return _validate_utc_millisecond_timestamp(value)

    def ensure_current(self, now: datetime) -> None:
        if (
            now.tzinfo is None
            or now.utcoffset() is None
            or now >= datetime.fromisoformat(self.expires_at)
        ):
            raise RuntimeError("Worker generation expired or clock unavailable")


def load_worker_generation(
    path: Path, camera_id: UUID, *, now: datetime
) -> WorkerGenerationCredential:
    try:
        if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
            raise ValueError()
        original = path.lstat()
        if not stat.S_ISREG(original.st_mode):
            raise ValueError()
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(descriptor, "rb") as file:
            metadata = os.fstat(file.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > 4096
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino) != (original.st_dev, original.st_ino)
            ):
                raise ValueError()
            if os.name != "nt" and (metadata.st_mode & 0o077 or metadata.st_uid != os.getuid()):
                raise ValueError()
            payload = file.read(4097)
            if len(payload) > 4096:
                raise ValueError()

            def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
                fields = dict(pairs)
                if len(fields) != len(pairs):
                    raise ValueError()
                return fields

            binding = WorkerGenerationCredential.model_validate(
                json.loads(payload, object_pairs_hook=unique_fields)
            )
        if binding.camera_id != str(camera_id):
            raise ValueError()
        binding.ensure_current(now)
        return binding
    except (OSError, ValueError, RuntimeError):
        raise ValueError("Worker generation credential is unavailable or invalid") from None


class GenerationBoundSource:
    """Bind one connection only; trusted renewal is required before reconnect."""

    def __init__(
        self,
        source: FrameSource,
        binding: WorkerGenerationCredential,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._source = source
        self._binding = binding
        self._clock = clock
        self._opened = False
        self._underlying_session: UUID | None = None

    @property
    def source_id(self) -> str:
        return self._source.source_id

    @property
    def is_connected(self) -> bool:
        return self._source.is_connected

    @property
    def binding(self) -> WorkerGenerationCredential:
        return self._binding

    def ensure_current(self) -> None:
        self._binding.ensure_current(self._clock())

    @property
    def is_ready_for_ack(self) -> bool:
        self.ensure_current()
        return self.is_connected and self._underlying_session is not None

    async def connect(self) -> None:
        self.ensure_current()
        if self._opened:
            raise RuntimeError("Worker generation requires trusted renewal before reconnect")
        self._opened = True
        await self._source.connect()

    async def read_frame(self) -> FrameEnvelope | None:
        self.ensure_current()
        frame = await self._source.read_frame()
        self.ensure_current()
        if frame is None:
            return None
        if self._underlying_session is None:
            if frame.sequence_number != 0:
                raise RuntimeError("Worker generation source must begin at sequence zero")
            self._underlying_session = frame.session_id
        elif frame.session_id != self._underlying_session:
            raise RuntimeError("Worker generation source session changed")
        return frame.model_copy(update={"session_id": UUID(self._binding.stream_session_id)})

    async def close(self) -> None:
        await self._source.close()
