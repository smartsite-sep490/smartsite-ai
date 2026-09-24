"""Strict Pydantic models for evaluation dataset manifests and ground-truth annotations."""

import math
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _coerce_tuple(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(v)
    return v


AnnotationsTuple = Annotated[tuple["GroundTruthObject", ...], BeforeValidator(_coerce_tuple)]

CANONICAL_PPE_CLASSES: tuple[str, ...] = (
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
)


class StrictEvaluationModel(BaseModel):
    """Base model enforcing strict types, forbidden extras, immutability, and no explicit nulls."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, val in data.items():
                if val is None:
                    raise ValueError(f"Explicit null is not permitted for property '{key}'")
        return data


class EvaluationBoundingBox(StrictEvaluationModel):
    """Normalized bounding box coordinates in [0.0, 1.0]."""

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_box(self) -> "EvaluationBoundingBox":
        for coord_name, coord in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
        ):
            if not math.isfinite(coord):
                raise ValueError(f"{coord_name} must be a finite number")
        if self.x1 >= self.x2:
            raise ValueError(f"x1 ({self.x1}) must be strictly less than x2 ({self.x2})")
        if self.y1 >= self.y2:
            raise ValueError(f"y1 ({self.y1}) must be strictly less than y2 ({self.y2})")
        return self


class GroundTruthObject(StrictEvaluationModel):
    """One annotated ground-truth object detection in a frame."""

    annotation_id: str = Field(..., alias="annotationId", min_length=1, max_length=128)
    class_name: str = Field(..., alias="className", min_length=1, max_length=64)
    bounding_box: EvaluationBoundingBox = Field(..., alias="boundingBox")
    person_instance_id: int | str | None = Field(default=None, alias="personInstanceId")
    related_person_annotation_id: str | None = Field(
        default=None, alias="relatedPersonAnnotationId", min_length=1, max_length=128
    )
    visibility: float = Field(default=1.0, alias="visibility", ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_object(self) -> "GroundTruthObject":
        if not math.isfinite(self.visibility):
            raise ValueError("visibility must be a finite number")
        return self


class EvaluationFrame(StrictEvaluationModel):
    """One annotated frame record in a dataset split index."""

    frame_id: str = Field(..., alias="frameId", min_length=1, max_length=128)
    media_path: str = Field(..., alias="mediaPath", min_length=1, max_length=512)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    width: int = Field(..., gt=0, le=32768)
    height: int = Field(..., gt=0, le=32768)
    frame_index: int | None = Field(default=None, alias="frameIndex", ge=0)
    video_time_seconds: float | None = Field(default=None, alias="videoTimeSeconds", ge=0.0)
    annotations: AnnotationsTuple = Field(default=(), alias="annotations")

    @model_validator(mode="after")
    def validate_frame(self) -> "EvaluationFrame":
        if (self.frame_index is None) != (self.video_time_seconds is None):
            raise ValueError(
                "Image frames must omit both frameIndex and videoTimeSeconds; "
                "video frames require both"
            )
        if self.video_time_seconds is not None and not math.isfinite(self.video_time_seconds):
            raise ValueError("videoTimeSeconds must be a finite number")
        return self


class GroundTruthPpeEpisode(StrictEvaluationModel):
    """Ground-truth temporal episode representing a continuous missing-PPE incident."""

    clip_id: str = Field(..., alias="clipId", min_length=1, max_length=128)
    person_instance_id: int | str = Field(..., alias="personInstanceId")
    ppe_item: str = Field(..., alias="ppeItem", min_length=1, max_length=64)
    start_time_seconds: float = Field(..., alias="startTimeSeconds", ge=0.0)
    end_time_seconds: float = Field(..., alias="endTimeSeconds", ge=0.0)

    @model_validator(mode="after")
    def validate_episode(self) -> "GroundTruthPpeEpisode":
        if not math.isfinite(self.start_time_seconds) or not math.isfinite(self.end_time_seconds):
            raise ValueError("startTimeSeconds and endTimeSeconds must be finite numbers")
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError(
                f"endTimeSeconds must be greater than or equal to startTimeSeconds: "
                f"{self.end_time_seconds} < {self.start_time_seconds}"
            )
        return self


class EvaluationManifest(StrictEvaluationModel):
    """Root dataset manifest declaring identity, license, checksum, class map, and splits."""

    schema_version: str = Field(..., alias="schemaVersion", pattern=r"^\d+\.\d+\.\d+$")
    dataset_id: str = Field(..., alias="datasetId", min_length=1, max_length=128)
    dataset_version: str = Field(..., alias="datasetVersion", min_length=1, max_length=64)
    source_url: str = Field(..., alias="sourceUrl", min_length=1, max_length=2048)
    license: str = Field(..., alias="license", min_length=1, max_length=128)
    aggregate_sha256: str = Field(..., alias="aggregateSha256", pattern=r"^[0-9a-f]{64}$")
    class_map: dict[str, str] = Field(..., alias="classMap", min_length=1)
    splits: dict[str, str] = Field(..., alias="splits", min_length=1)
