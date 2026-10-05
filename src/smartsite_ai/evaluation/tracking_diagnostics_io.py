"""Strict loader for a reviewed matched-ledger diagnostic file.

The file is a human-reviewed association ledger. It is not a raw-media corpus,
a detection matcher, or evidence that a tracker benchmark has passed.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from smartsite_ai.evaluation.tracking_continuity import TrackingMatch

_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_MATCHES = 20_000
_MAX_IDENTIFIER_LENGTH = 128
_MAX_RIGHTS_LENGTH = 1000
_MAX_TIMESTAMP_LENGTH = 64
_MAX_FRAME_INDEX = 2**53 - 1
_PATH_ERROR = "tracking diagnostics path is not a regular file"
_SIZE_ERROR = "tracking diagnostics file exceeds the size limit"
_DOCUMENT_ERROR = "tracking diagnostics document is invalid"


class TrackingDiagnosticsError(ValueError):
    """Loader failure whose message never includes the input or its path."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _MatchRecord(_StrictModel):
    camera_id: str = Field(alias="cameraId")
    clip_id: str = Field(alias="clipId")
    session_id: str = Field(alias="streamSessionId")
    frame_index: int = Field(alias="frameIndex", ge=0, le=_MAX_FRAME_INDEX)
    person_key: str = Field(alias="groundTruthPersonKey")
    track_id: str | None = Field(alias="predictedTrackId")

    @field_validator("camera_id", "clip_id", "session_id", "person_key")
    @classmethod
    def identifiers_are_bounded(cls, value: str) -> str:
        _require_unpadded(value, _MAX_IDENTIFIER_LENGTH)
        return value

    @field_validator("track_id")
    @classmethod
    def track_is_bounded(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _require_unpadded(value, _MAX_IDENTIFIER_LENGTH)
        return value


class _LedgerDocument(_StrictModel):
    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    purpose: Literal["TRACKING_ASSOCIATION_DIAGNOSTICS"] = Field(alias="purpose")
    reviewed_by: str = Field(alias="reviewedBy")
    reviewed_at_utc: str = Field(alias="reviewedAtUtc")
    source_rights: str = Field(alias="sourceRights")
    matches: list[_MatchRecord] = Field(max_length=_MAX_MATCHES)

    @field_validator("reviewed_by")
    @classmethod
    def reviewer_is_bounded(cls, value: str) -> str:
        _require_unpadded(value, _MAX_IDENTIFIER_LENGTH)
        return value

    @field_validator("source_rights")
    @classmethod
    def rights_are_bounded(cls, value: str) -> str:
        _require_unpadded(value, _MAX_RIGHTS_LENGTH)
        return value

    @field_validator("reviewed_at_utc")
    @classmethod
    def timestamp_is_utc(cls, value: str) -> str:
        _require_utc(value)
        return value


def load_tracking_diagnostics(path: Path) -> tuple[TrackingMatch, ...]:
    """Load reviewed matches from one regular local JSON file."""

    raw = _read_bounded_file(path)
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
        document = _LedgerDocument.model_validate(payload)
        return tuple(
            TrackingMatch(
                camera_id=item.camera_id,
                clip_id=item.clip_id,
                session_id=item.session_id,
                frame_index=item.frame_index,
                person_key=item.person_key,
                track_id=item.track_id,
            )
            for item in document.matches
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        TypeError,
        RecursionError,
    ):
        raise TrackingDiagnosticsError(_DOCUMENT_ERROR) from None


def _read_bounded_file(path: Path) -> bytes:
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for index, part in enumerate(absolute.parts[1:]):
            current /= part
            info = current.lstat()
            if _is_link_like(current, info):
                raise TrackingDiagnosticsError(_PATH_ERROR)
            is_last = index == len(absolute.parts[1:]) - 1
            if is_last and not stat.S_ISREG(info.st_mode):
                raise TrackingDiagnosticsError(_PATH_ERROR)
            if is_last and info.st_size > _MAX_FILE_BYTES:
                raise TrackingDiagnosticsError(_SIZE_ERROR)
    except TrackingDiagnosticsError:
        raise
    except (OSError, ValueError):
        raise TrackingDiagnosticsError(_PATH_ERROR) from None

    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except (OSError, ValueError):
        raise TrackingDiagnosticsError(_PATH_ERROR) from None
    if len(raw) > _MAX_FILE_BYTES:
        raise TrackingDiagnosticsError(_SIZE_ERROR)
    return raw


def _is_link_like(path: Path, info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    is_junction = bool(getattr(os.path, "isjunction", lambda _value: False)(path))
    return (
        stat.S_ISLNK(info.st_mode)
        or path.is_symlink()
        or is_junction
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_number(constant: str) -> None:
    del constant
    raise ValueError("nonstandard JSON number")


def _require_unpadded(value: str, limit: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > limit
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("invalid bounded text")


def _require_utc(value: str) -> None:
    _require_unpadded(value, _MAX_TIMESTAMP_LENGTH)
    text = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("invalid timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("invalid timestamp")


__all__ = ["TrackingDiagnosticsError", "load_tracking_diagnostics"]
