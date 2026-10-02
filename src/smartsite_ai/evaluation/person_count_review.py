"""Explicit full-frame count review, bound to immutable evaluation inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from smartsite_ai.evaluation.dataset import MAX_FRAMES_PER_SPLIT
from smartsite_ai.evaluation.models import MAX_IDENTIFIER_LENGTH, MAX_SAFE_INTEGER, EvaluationFrame

MAX_COUNT_REVIEW_BYTES = 16 * 1024 * 1024


def _datetime_from_iso(value: object) -> object:
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


class PersonCountReview(BaseModel):
    """A review assertion, not proof that a human performed or approved it."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    frame_id: str = Field(alias="frameId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    media_sha256: str = Field(alias="mediaSha256", pattern=r"^[0-9a-f]{64}$")
    review_scope: Literal["FULL_FRAME_PERSON_COUNT"] = Field(alias="reviewScope")
    review_status: Literal["REVIEWED", "EXCLUDED"] = Field(alias="reviewStatus")
    ground_truth_count: int | None = Field(
        default=None, alias="groundTruthCount", ge=0, le=MAX_SAFE_INTEGER
    )
    reviewed_by: str = Field(alias="reviewedBy", min_length=1, max_length=128)
    reviewed_at_utc: Annotated[datetime, BeforeValidator(_datetime_from_iso)] = Field(
        alias="reviewedAtUtc"
    )
    reason: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("frame_id", "reviewed_by", "reason")
    @classmethod
    def reject_blank_padded_text(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("review text must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if self.reviewed_at_utc.utcoffset() != timedelta(0):
            raise ValueError("reviewedAtUtc must be timezone-aware UTC")
        if (self.review_status == "REVIEWED") != (self.ground_truth_count is not None):
            raise ValueError("only REVIEWED counts must declare groundTruthCount")
        if self.review_status == "EXCLUDED" and self.reason is None:
            raise ValueError("EXCLUDED frame must declare reason")
        return self


class _ReviewIndex(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: Literal["1.0.0"] = Field(alias="schemaVersion")
    dataset_aggregate_sha256: str = Field(alias="datasetAggregateSha256", pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]
    frames: Annotated[
        tuple[PersonCountReview, ...],
        BeforeValidator(lambda value: tuple(value) if isinstance(value, list) else value),
    ] = Field(max_length=MAX_FRAMES_PER_SPLIT)


@dataclass(frozen=True, slots=True)
class ValidatedCountReviewIndex:
    sha256: str
    frames: Mapping[str, PersonCountReview]


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in person count review index")
        result[key] = value
    return result


def load_person_count_review_index(
    path: Path,
    *,
    frames: Sequence[EvaluationFrame],
    dataset_aggregate_sha256: str,
    split: Literal["train", "validation", "test"],
) -> ValidatedCountReviewIndex:
    """Validate once before inference; never infer review from annotations or predictions."""

    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_COUNT_REVIEW_BYTES + 1)
        if len(content) > MAX_COUNT_REVIEW_BYTES:
            raise ValueError("person count review index exceeds size limit")
        index = _ReviewIndex.model_validate(
            json.loads(content.decode("utf-8"), object_pairs_hook=_unique_keys)
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("invalid person count review index") from error
    if index.dataset_aggregate_sha256 != dataset_aggregate_sha256 or index.split != split:
        raise ValueError("person count review does not match dataset/split")
    frames_by_id = {frame.frame_id: frame for frame in frames}
    reviewed: dict[str, PersonCountReview] = {}
    for review in index.frames:
        if review.frame_id in reviewed:
            raise ValueError("duplicate frameId in person count review")
        frame = frames_by_id.get(review.frame_id)
        if frame is None or frame.sha256 != review.media_sha256:
            raise ValueError("person count review frame/media does not match evaluated split")
        reviewed[review.frame_id] = review
    return ValidatedCountReviewIndex(
        hashlib.sha256(content).hexdigest(), MappingProxyType(reviewed)
    )
