"""Strict Pydantic models for evaluation dataset manifests and ground-truth annotations."""

import math
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

MAX_ANNOTATIONS_PER_FRAME: int = 500
MAX_IDENTIFIER_LENGTH: int = 128
MAX_PATH_LENGTH: int = 512
MAX_URL_LENGTH: int = 2048
MAX_FRAME_DIMENSION: int = 32768
MAX_FRAME_INDEX: int = 10_000_000
MAX_VIDEO_TIME_SECONDS: float = 86400.0 * 7.0
MAX_SAFE_INTEGER: int = 9_007_199_254_740_991
MAX_SOURCE_CLASS_NAME_LENGTH: int = 64

CANONICAL_PPE_CLASSES: tuple[str, ...] = (
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
)
CANONICAL_PPE_CLASSES_SET: frozenset[str] = frozenset(CANONICAL_PPE_CLASSES)
CanonicalPpeClassName = Literal[
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
]

REQUIRED_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
REQUIRED_SPLIT_NAMES_SET: frozenset[str] = frozenset(REQUIRED_SPLIT_NAMES)

VALID_PPE_ITEMS: tuple[str, ...] = ("HARD_HAT", "SAFETY_VEST")


def _coerce_tuple(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(v)
    return v


AnnotationsTuple = Annotated[tuple["GroundTruthObject", ...], BeforeValidator(_coerce_tuple)]
ObservablePpeItemsTuple = Annotated[
    tuple[Literal["HARD_HAT", "SAFETY_VEST"], ...], BeforeValidator(_coerce_tuple)
]
ImmutableStringMapping = Annotated[
    Mapping[str, str],
    PlainSerializer(lambda value: dict(value), return_type=dict[str, str]),
]


class FrozenStringMap(Mapping[str, str]):
    """Small immutable, hashable string mapping with safe deepcopy semantics."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, str]) -> None:
        object.__setattr__(self, "_items", tuple(sorted(values.items())))

    def __getitem__(self, key: str) -> str:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenStringMap":
        del memo
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("FrozenStringMap is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("FrozenStringMap is immutable")


def _validate_person_instance_id(v: Any) -> int | str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("boolean is not a valid personInstanceId")
    if isinstance(v, int):
        if v < 0 or v > MAX_SAFE_INTEGER:
            raise ValueError(f"Integer personInstanceId must be between 0 and {MAX_SAFE_INTEGER}")
        return v
    if isinstance(v, str):
        if len(v) < 1 or len(v) > 64 or not v.strip() or v != v.strip():
            raise ValueError("String personInstanceId must be between 1 and 64 characters")
        return v
    raise ValueError("personInstanceId must be an integer, string, or None")


PersonInstanceId = Annotated[int | str | None, BeforeValidator(_validate_person_instance_id)]


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

    @model_serializer(mode="wrap")
    def serialize_without_explicit_nulls(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        serialized = handler(self)
        return {key: value for key, value in serialized.items() if value is not None}


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

    annotation_id: str = Field(
        ..., alias="annotationId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH
    )
    class_name: CanonicalPpeClassName = Field(..., alias="className")
    bounding_box: EvaluationBoundingBox = Field(..., alias="boundingBox")
    person_instance_id: PersonInstanceId = Field(default=None, alias="personInstanceId")
    related_person_annotation_id: str | None = Field(
        default=None,
        alias="relatedPersonAnnotationId",
        min_length=1,
        max_length=MAX_IDENTIFIER_LENGTH,
    )
    visibility: float = Field(default=1.0, alias="visibility", ge=0.0, le=1.0)
    observable_ppe_items: ObservablePpeItemsTuple | None = Field(
        default=None,
        alias="observablePpeItems",
        max_length=len(VALID_PPE_ITEMS),
    )

    @field_validator("annotation_id", "related_person_annotation_id")
    @classmethod
    def validate_annotation_identifiers(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("annotation identifiers must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_object(self) -> "GroundTruthObject":
        if not math.isfinite(self.visibility):
            raise ValueError("visibility must be a finite number")
        if self.observable_ppe_items is not None:
            if self.class_name != "Person":
                raise ValueError("observablePpeItems is only valid for Person annotations")
            if len(set(self.observable_ppe_items)) != len(self.observable_ppe_items):
                raise ValueError("observablePpeItems must not contain duplicates")
            canonical_items = tuple(
                item for item in VALID_PPE_ITEMS if item in self.observable_ppe_items
            )
            object.__setattr__(self, "observable_ppe_items", canonical_items)
        if self.class_name == "Person" and self.related_person_annotation_id is not None:
            raise ValueError("Person annotations must not declare relatedPersonAnnotationId")
        return self


class EvaluationFrame(StrictEvaluationModel):
    """One annotated frame record in a dataset split index."""

    frame_id: str = Field(..., alias="frameId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    media_path: str = Field(..., alias="mediaPath", min_length=1, max_length=MAX_PATH_LENGTH)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    width: int = Field(..., gt=0, le=MAX_FRAME_DIMENSION)
    height: int = Field(..., gt=0, le=MAX_FRAME_DIMENSION)
    frame_index: int | None = Field(default=None, alias="frameIndex", ge=0, le=MAX_FRAME_INDEX)
    video_time_seconds: float | None = Field(
        default=None, alias="videoTimeSeconds", ge=0.0, le=MAX_VIDEO_TIME_SECONDS
    )
    annotations: AnnotationsTuple = Field(
        default=(), alias="annotations", max_length=MAX_ANNOTATIONS_PER_FRAME
    )

    @field_validator("frame_id", "media_path")
    @classmethod
    def validate_frame_identifiers_and_path(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("frameId and mediaPath must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_frame(self) -> "EvaluationFrame":
        suffix = PurePosixPath(self.media_path.replace("\\", "/")).suffix.lower()
        image_suffixes = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
        video_suffixes = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
        if suffix not in image_suffixes | video_suffixes:
            raise ValueError(f"mediaPath has unsupported image/video extension: '{suffix}'")
        has_video_position = self.frame_index is not None and self.video_time_seconds is not None
        if (self.frame_index is None) != (self.video_time_seconds is None):
            raise ValueError(
                "Image frames must omit both frameIndex and videoTimeSeconds; "
                "video frames require both"
            )
        if suffix in image_suffixes and has_video_position:
            raise ValueError("Image mediaPath must omit frameIndex and videoTimeSeconds")
        if suffix in video_suffixes and not has_video_position:
            raise ValueError("Video mediaPath requires frameIndex and videoTimeSeconds")
        if self.video_time_seconds is not None and not math.isfinite(self.video_time_seconds):
            raise ValueError("videoTimeSeconds must be a finite number")
        if len(self.annotations) > MAX_ANNOTATIONS_PER_FRAME:
            raise ValueError(
                f"Number of annotations ({len(self.annotations)}) exceeds maximum "
                f"allowed ({MAX_ANNOTATIONS_PER_FRAME})"
            )
        return self


class GroundTruthPpeEpisode(StrictEvaluationModel):
    """Ground-truth temporal episode representing a continuous missing-PPE incident."""

    clip_id: str = Field(..., alias="clipId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    person_instance_id: PersonInstanceId = Field(..., alias="personInstanceId")
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"] = Field(..., alias="ppeItem")
    start_time_seconds: float = Field(
        ..., alias="startTimeSeconds", ge=0.0, le=MAX_VIDEO_TIME_SECONDS
    )
    end_time_seconds: float = Field(..., alias="endTimeSeconds", ge=0.0, le=MAX_VIDEO_TIME_SECONDS)

    @field_validator("clip_id")
    @classmethod
    def validate_clip_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("clipId must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_episode(self) -> "GroundTruthPpeEpisode":
        if self.person_instance_id is None:
            raise ValueError("personInstanceId is required for GroundTruthPpeEpisode")
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

    schema_version: Literal["1.0.0"] = Field(..., alias="schemaVersion")
    dataset_id: str = Field(..., alias="datasetId", min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    dataset_version: str = Field(..., alias="datasetVersion", min_length=1, max_length=64)
    source_url: str = Field(..., alias="sourceUrl", min_length=1, max_length=MAX_URL_LENGTH)
    license: str = Field(..., alias="license", min_length=1, max_length=128)
    aggregate_sha256: str = Field(..., alias="aggregateSha256", pattern=r"^[0-9a-f]{64}$")
    class_map: ImmutableStringMapping = Field(..., alias="classMap", min_length=1)
    splits: ImmutableStringMapping = Field(..., alias="splits", min_length=1)

    @field_validator("dataset_id", "dataset_version", "license")
    @classmethod
    def reject_blank_or_padded_strings(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("value must be non-blank and must not have surrounding whitespace")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("sourceUrl must not have surrounding whitespace")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("sourceUrl must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("sourceUrl must not contain credentials")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> "EvaluationManifest":
        for source_class, canonical_class in self.class_map.items():
            if not isinstance(source_class, str) or not isinstance(canonical_class, str):
                raise ValueError("classMap keys and values must be strings")
            if (
                not source_class.strip()
                or source_class != source_class.strip()
                or len(source_class) > MAX_SOURCE_CLASS_NAME_LENGTH
            ):
                raise ValueError(
                    f"classMap source names must be 1..{MAX_SOURCE_CLASS_NAME_LENGTH} characters "
                    "without surrounding whitespace"
                )

        # Validate canonical classes
        classes_in_map = set(self.class_map.values())
        if classes_in_map != CANONICAL_PPE_CLASSES_SET or len(self.class_map) != len(
            CANONICAL_PPE_CLASSES
        ):
            raise ValueError(
                f"Canonical class map must contain exactly the 5 classes: "
                f"{sorted(CANONICAL_PPE_CLASSES_SET)} with no duplicates; "
                f"found {sorted(classes_in_map)}"
            )

        # Validate split names
        splits_keys = set(self.splits.keys())
        if splits_keys != REQUIRED_SPLIT_NAMES_SET:
            raise ValueError(
                f"Splits must contain exactly {sorted(REQUIRED_SPLIT_NAMES_SET)}; "
                f"found {sorted(splits_keys)}"
            )

        for split_name, split_path in self.splits.items():
            if not isinstance(split_path, str):
                raise ValueError(f"Split '{split_name}' path must be a string")
            if (
                not split_path.strip()
                or split_path != split_path.strip()
                or len(split_path) > MAX_PATH_LENGTH
            ):
                raise ValueError(
                    f"Split '{split_name}' path must be 1..{MAX_PATH_LENGTH} characters "
                    "without surrounding whitespace"
                )

        object.__setattr__(self, "class_map", FrozenStringMap(self.class_map))
        object.__setattr__(self, "splits", FrozenStringMap(self.splits))
        return self
