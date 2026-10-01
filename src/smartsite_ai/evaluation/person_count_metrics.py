"""Frame-level count errors against explicit reviewed counts, never unique Workers."""

import math
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartsite_ai.evaluation.models import MAX_IDENTIFIER_LENGTH, MAX_SAFE_INTEGER


class PersonCountSample(BaseModel):
    """The caller supplies reviewed truth; predictions never supply missing truth."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    frame_id: str = Field(alias="frameId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    predicted_count: int = Field(alias="predictedCount", ge=0, le=MAX_SAFE_INTEGER)
    ground_truth_count: int | None = Field(alias="groundTruthCount", ge=0, le=MAX_SAFE_INTEGER)
    review_status: Literal["REVIEWED", "UNREVIEWED", "EXCLUDED"] = Field(alias="reviewStatus")

    @field_validator("frame_id")
    @classmethod
    def validate_frame_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("frameId must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_reviewed_truth(self) -> Self:
        if (self.review_status == "REVIEWED") != (self.ground_truth_count is not None):
            raise ValueError("only reviewed frames must have a groundTruthCount")
        return self


class PersonCountReport(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    evaluated_frames: int
    mean_absolute_error: float | None
    root_mean_squared_error: float | None
    mean_signed_error: float | None
    exact_match_fraction: float | None
    overcount_frames: int
    undercount_frames: int
    ground_truth_people_total: int
    predicted_people_total: int
    unreviewed_frame_ids: tuple[str, ...]
    excluded_frame_ids: tuple[str, ...]
    reason: str | None
    scope: Literal["REVIEWED_FRAME_COUNTS_ONLY"] = "REVIEWED_FRAME_COUNTS_ONLY"
    is_model_acceptance: Literal[False] = False
    warning: str = (
        "Totals sum frame counts, not unique Workers. Count accuracy does not measure "
        "localization, tracking, identity or PPE association. REVIEWED is a caller assertion; "
        "this computation does not verify human review or approve a model/dataset."
    )


def compute_person_count_metrics(samples: Sequence[PersonCountSample]) -> PersonCountReport:
    seen: set[str] = set()
    errors: list[int] = []
    unreviewed: list[str] = []
    excluded: list[str] = []
    actual_total = predicted_total = 0
    for sample in samples:
        if sample.frame_id in seen:
            raise ValueError(f"duplicate frameId: {sample.frame_id}")
        seen.add(sample.frame_id)
        if sample.review_status == "UNREVIEWED":
            unreviewed.append(sample.frame_id)
        elif sample.review_status == "EXCLUDED":
            excluded.append(sample.frame_id)
        else:
            assert sample.ground_truth_count is not None
            errors.append(sample.predicted_count - sample.ground_truth_count)
            actual_total += sample.ground_truth_count
            predicted_total += sample.predicted_count

    count = len(errors)
    return PersonCountReport(
        evaluated_frames=count,
        mean_absolute_error=sum(abs(error) for error in errors) / count if count else None,
        root_mean_squared_error=(
            math.sqrt(sum(error * error for error in errors) / count) if count else None
        ),
        mean_signed_error=sum(errors) / count if count else None,
        exact_match_fraction=sum(error == 0 for error in errors) / count if count else None,
        overcount_frames=sum(error > 0 for error in errors),
        undercount_frames=sum(error < 0 for error in errors),
        ground_truth_people_total=actual_total,
        predicted_people_total=predicted_total,
        unreviewed_frame_ids=tuple(sorted(unreviewed)),
        excluded_frame_ids=tuple(sorted(excluded)),
        reason=None if count else "no reviewed frame counts",
    )
