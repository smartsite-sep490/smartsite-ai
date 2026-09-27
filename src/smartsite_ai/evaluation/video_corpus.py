"""Reviewed temporal-video corpus contracts for MF05 event-level evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BeforeValidator, Field, field_validator, model_validator

from smartsite_ai.evaluation.models import (
    MAX_FRAME_DIMENSION,
    MAX_FRAME_INDEX,
    MAX_IDENTIFIER_LENGTH,
    MAX_VIDEO_TIME_SECONDS,
    GroundTruthObject,
    GroundTruthPpeEpisode,
    StrictEvaluationModel,
)

MAX_CLIPS = 256
MAX_REVIEWER_LENGTH = 128


def _coerce_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _coerce_path(value: object) -> object:
    return Path(value) if isinstance(value, str) else value


def _coerce_datetime(value: object) -> object:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value


InputPath = Annotated[Path, BeforeValidator(_coerce_path)]
InputDateTime = Annotated[datetime, BeforeValidator(_coerce_datetime)]
ReviewedAnnotations = Annotated[tuple[GroundTruthObject, ...], BeforeValidator(_coerce_tuple)]


class ReviewedClip(StrictEvaluationModel):
    """One immutable local video and its reviewed frame-label index."""

    clip_id: str = Field(alias="clipId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    media_path: InputPath = Field(alias="mediaPath")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frames_path: InputPath = Field(alias="framesPath")

    @field_validator("clip_id")
    @classmethod
    def validate_clip_id(cls, value: str) -> str:
        if value != value.strip() or not value.strip():
            raise ValueError("clipId must be non-blank and unpadded")
        if not all(character.isalnum() or character in {"-", "_"} for character in value):
            raise ValueError("clipId may contain only letters, digits, hyphen, and underscore")
        return value

    @field_validator("media_path", "frames_path")
    @classmethod
    def require_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("reviewed input paths must be absolute")
        return value


ReviewedClipsTuple = Annotated[tuple[ReviewedClip, ...], BeforeValidator(_coerce_tuple)]


class ReviewedCorpusSource(StrictEvaluationModel):
    """Human-reviewed inputs from which the publishable evaluation corpus is built."""

    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    dataset_version: str = Field(alias="datasetVersion", min_length=1, max_length=64)
    source_url: str = Field(alias="sourceUrl", min_length=1, max_length=2048)
    license: str = Field(min_length=1, max_length=128)
    reviewed_by: str = Field(alias="reviewedBy", min_length=1, max_length=MAX_REVIEWER_LENGTH)
    reviewed_at_utc: InputDateTime = Field(alias="reviewedAtUtc")
    max_frame_gap_seconds: float = Field(alias="maxFrameGapSeconds", gt=0.0, le=1.0)
    episodes_path: InputPath = Field(alias="episodesPath")
    clips: ReviewedClipsTuple = Field(min_length=1, max_length=MAX_CLIPS)

    @field_validator("dataset_id", "dataset_version", "license", "reviewed_by")
    @classmethod
    def reject_padded_text(cls, value: str) -> str:
        if value != value.strip() or not value.strip():
            raise ValueError("review metadata values must be non-blank and unpadded")
        return value

    @field_validator("episodes_path")
    @classmethod
    def require_absolute_episode_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("episodesPath must be absolute")
        return value

    @model_validator(mode="after")
    def validate_unique_clips_and_review_time(self) -> ReviewedCorpusSource:
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clipId values must be unique")
        if self.reviewed_at_utc.tzinfo is None or self.reviewed_at_utc.utcoffset() is None:
            raise ValueError("reviewedAtUtc must include a UTC offset")
        if self.reviewed_at_utc.utcoffset().total_seconds() != 0:
            raise ValueError("reviewedAtUtc must use UTC")
        return self


class ReviewedFrame(StrictEvaluationModel):
    """One manually reviewed video frame before media identity is attached."""

    clip_id: str = Field(alias="clipId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    frame_index: int = Field(alias="frameIndex", ge=0, le=MAX_FRAME_INDEX)
    video_time_seconds: float = Field(alias="videoTimeSeconds", ge=0.0, le=MAX_VIDEO_TIME_SECONDS)
    width: int = Field(gt=0, le=MAX_FRAME_DIMENSION)
    height: int = Field(gt=0, le=MAX_FRAME_DIMENSION)
    annotations: ReviewedAnnotations = Field(default=(), max_length=500)

    @field_validator("clip_id")
    @classmethod
    def reject_blank_clip_id(cls, value: str) -> str:
        if value != value.strip() or not value.strip():
            raise ValueError("clipId must be non-blank and unpadded")
        return value

    @field_validator("video_time_seconds")
    @classmethod
    def require_finite_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("videoTimeSeconds must be finite")
        return value


@dataclass(frozen=True, slots=True)
class VideoFacts:
    """Container facts obtained without decoding model inputs."""

    width: int
    height: int
    frame_count: int
    duration_seconds: float


class VideoProbe(Protocol):
    """Boundary used to verify labels against the actual local media."""

    def probe(self, path: Path) -> VideoFacts: ...

    def frame_times(self, path: Path, frame_indexes: Sequence[int]) -> Mapping[int, float]: ...


class OpenCvVideoProbe:
    """Lazy OpenCV metadata probe used by the local corpus builder."""

    def __init__(self, cv2_loader: Callable[[], object] | None = None) -> None:
        self._cv2_loader = cv2_loader or self._load_cv2

    @staticmethod
    def _load_cv2() -> object:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is unavailable; run the corpus builder with the vision or cuda126 extra"
            ) from error
        return cv2

    def probe(self, path: Path) -> VideoFacts:
        cv2 = self._cv2_loader()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV could not open reviewed video")
            width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if width <= 0 or height <= 0 or frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("reviewed video metadata is incomplete or invalid")
        duration = frame_count / fps
        if not math.isfinite(duration) or duration <= 0 or duration > MAX_VIDEO_TIME_SECONDS:
            raise ValueError("reviewed video duration is outside supported bounds")
        return VideoFacts(width, height, frame_count, duration)

    def frame_times(self, path: Path, frame_indexes: Sequence[int]) -> Mapping[int, float]:
        cv2 = self._cv2_loader()
        capture = cv2.VideoCapture(str(path))
        results: dict[int, float] = {}
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV could not open reviewed video")
            for frame_index in frame_indexes:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
                    raise ValueError(f"OpenCV could not seek reviewed frame {frame_index}")
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"OpenCV could not decode reviewed frame {frame_index}")
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError(
                        f"OpenCV returned invalid time for reviewed frame {frame_index}"
                    )
                results[frame_index] = timestamp
        finally:
            capture.release()
        return results


def source_episode_key(episode: GroundTruthPpeEpisode) -> tuple[str, int | str, str]:
    """Return the identity of one labelled temporal PPE series."""

    assert episode.person_instance_id is not None
    return episode.clip_id, episode.person_instance_id, episode.ppe_item


def evaluated_camera_seconds(frames_by_clip: Mapping[str, Sequence[ReviewedFrame]]) -> float:
    """Use the same sampled interval definition as the official evaluator."""

    return sum(
        max(frame.video_time_seconds for frame in frames)
        - min(frame.video_time_seconds for frame in frames)
        for frames in frames_by_clip.values()
        if frames
    )


__all__ = [
    "MAX_CLIPS",
    "OpenCvVideoProbe",
    "ReviewedCorpusSource",
    "ReviewedFrame",
    "VideoFacts",
    "VideoProbe",
    "evaluated_camera_seconds",
    "source_episode_key",
]
