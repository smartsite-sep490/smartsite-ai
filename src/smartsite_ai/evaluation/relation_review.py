"""Offline relation proposals; coordinate review never establishes wearing truth."""

import hashlib
import io
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_REVIEW_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Dimension = Annotated[int, Field(gt=0, le=32768)]
Box = tuple[float, float, float, float]
BodyPart = Literal["HEAD", "TORSO", "HANDS", "FEET", "EYES"]
_PARTS = {1: "HEAD", 3: "TORSO", 5: "HANDS", 7: "FEET", 9: "EYES"}
# Versioned review taxonomy, independent of opt-in runtime profile changes.
_CLASS_NAMES = {
    0: "Person",
    1: "Hardhat",
    3: "Safety Vest",
    5: "Gloves",
    7: "Boots",
    9: "Goggles",
    11: "Mask",
    13: "fullbody-Harness",
    14: "Lanyard-Hook",
}


class ReviewModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class ReviewObject(ReviewModel):
    id: Identifier
    class_id: int
    class_name: str
    box: Box

    @model_validator(mode="after")
    def validate_class(self) -> Self:
        if _CLASS_NAMES.get(self.class_id) != self.class_name:
            raise ValueError("positive unified class id/name required")
        return self


class RelationProposal(ReviewModel):
    object_id: Identifier
    person_id: Identifier | None
    body_part: BodyPart
    visibility: Literal["VISIBLE", "OCCLUDED_OR_CROPPED", "UNRESOLVED"]
    proposal: Literal["WORN_AT_EXPECTED_PART", "CARRIED_OR_HANGING", "UNRESOLVED"]
    body_part_box: Box | None

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        if self.person_id is None and (
            self.proposal != "UNRESOLVED"
            or self.visibility != "UNRESOLVED"
            or self.body_part_box is not None
        ):
            raise ValueError("unassigned object cannot claim a person relation or body part")
        if self.proposal == "WORN_AT_EXPECTED_PART" and (
            self.visibility != "VISIBLE" or self.body_part_box is None
        ):
            raise ValueError("worn proposal requires visible reviewed body-part coordinates")
        return self


def _check_box(box: Box, dimensions: tuple[int, int]) -> None:
    x1, y1, x2, y2 = box
    width, height = dimensions
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box outside display bounds or empty")


class RelationReview(ReviewModel):
    schema_version: Literal["smartsite.ppe-relation-review/1"]
    status: Literal["DRAFT_ONLY"]
    image_id: Identifier
    source_sha256: Sha256
    coordinate_receipt_sha256: Sha256
    source_group: Identifier
    encoded_dimensions: tuple[Dimension, Dimension]
    exif_orientation: int = Field(ge=1, le=8)
    display_dimensions: tuple[Dimension, Dimension]
    coordinate_space: Literal["EXIF_NORMALIZED_DISPLAY_PIXELS"]
    objects: tuple[ReviewObject, ...] = Field(min_length=1, max_length=500)
    relations: tuple[RelationProposal, ...] = Field(max_length=500)

    @model_validator(mode="after")
    def validate_links(self) -> Self:
        expected = self.encoded_dimensions
        if self.exif_orientation >= 5:
            expected = expected[::-1]
        if self.display_dimensions != expected:
            raise ValueError("EXIF/display dimensions disagree")
        objects = {obj.id: obj for obj in self.objects}
        if len(objects) != len(self.objects):
            raise ValueError("duplicate object id")
        for obj in self.objects:
            _check_box(obj.box, self.display_dimensions)
        linked = set()
        for relation in self.relations:
            obj = objects.get(relation.object_id)
            if obj is None or obj.class_id not in _PARTS:
                raise ValueError("relation must reference an observation-family object")
            if relation.object_id in linked:
                raise ValueError("duplicate object relation")
            linked.add(relation.object_id)
            if relation.body_part != _PARTS[obj.class_id]:
                raise ValueError("body part contradicts object class")
            if relation.person_id is not None:
                person = objects.get(relation.person_id)
                if person is None or person.class_id != 0:
                    raise ValueError("relation owner must reference a Person in this image")
            if relation.body_part_box is not None:
                _check_box(relation.body_part_box, self.display_dimensions)
        if linked != {obj.id for obj in self.objects if obj.class_id in _PARTS}:
            raise ValueError("each observation-family object needs an explicit review row")
        return self


def _read_bounded(path: Path, limit: int) -> bytes:
    if not path.is_file():
        raise ValueError("input must be a regular file")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("input exceeds size limit")
    return content


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def load_relation_review(path: Path) -> RelationReview:
    payload = _read_bounded(path, MAX_REVIEW_BYTES)
    json.loads(payload, object_pairs_hook=_unique_fields)
    return RelationReview.model_validate_json(payload)


def verify_source_image(review: RelationReview, path: Path) -> None:
    """Check the exact encoded bytes and orientation; never infer labels or rights."""
    payload = _read_bounded(path, MAX_IMAGE_BYTES)
    if hashlib.sha256(payload).hexdigest() != review.source_sha256:
        raise ValueError("source hash mismatch")
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != review.encoded_dimensions:
            raise ValueError("encoded dimensions mismatch")
        if image.getexif().get(274, 1) != review.exif_orientation:
            raise ValueError("source orientation mismatch")
        with ImageOps.exif_transpose(image) as display:
            if display.size != review.display_dimensions:
                raise ValueError("display dimensions mismatch")


def verify_coordinate_receipt(review: RelationReview, path: Path) -> None:
    """Bind a trusted local receipt's bytes; this cannot authenticate its human author."""
    payload = _read_bounded(path, MAX_REVIEW_BYTES)
    if hashlib.sha256(payload).hexdigest() != review.coordinate_receipt_sha256:
        raise ValueError("coordinate receipt hash mismatch")
    receipt = json.loads(payload, object_pairs_hook=_unique_fields)
    if not isinstance(receipt, dict):
        raise ValueError("coordinate receipt must be an object")
    images = receipt.get("images", [])
    if not isinstance(images, list) or not all(isinstance(image, dict) for image in images):
        raise ValueError("coordinate receipt images must be objects")
    matches = [image for image in images if image.get("id") == review.image_id]
    if len(matches) != 1 or not receipt.get("questionItemId") or not receipt.get("answer"):
        raise ValueError("coordinate receipt image/review provenance missing")
    image = matches[0]
    source_hash = image.get("sourceSha256", image.get("sha256"))
    dimensions = image.get("displayDimensions", image.get("dimensions"))
    if source_hash != review.source_sha256 or dimensions != list(review.display_dimensions):
        raise ValueError("coordinate receipt image binding mismatch")
    if not isinstance(dimensions, list) or any(type(n) is not int for n in dimensions):
        raise ValueError("coordinate receipt dimensions must be integers")
    if "encodedDimensions" in image and (
        image["encodedDimensions"] != list(review.encoded_dimensions)
        or any(type(n) is not int for n in image["encodedDimensions"])
    ):
        raise ValueError("coordinate receipt encoded dimensions mismatch")
    if "exifOrientation" in image and (
        type(image["exifOrientation"]) is not int
        or image["exifOrientation"] != review.exif_orientation
    ):
        raise ValueError("coordinate receipt orientation mismatch")
    coordinate_space = image.get("coordinateSpace")
    # Older C01/C02 receipts predate explicit display coordinates; only identity EXIF is safe.
    if coordinate_space != review.coordinate_space and not (
        coordinate_space is None and review.exif_orientation == 1
    ):
        raise ValueError("coordinate receipt frame mismatch")
    boxes = image.get("boxes", [])
    required = {"id", "classId", "className", "xyxyPixels"}
    if not isinstance(boxes, list) or not all(
        isinstance(box, dict) and required <= box.keys() and isinstance(box["id"], str)
        for box in boxes
    ):
        raise ValueError("coordinate receipt boxes malformed")
    expected = {obj.id: (obj.class_id, obj.class_name, list(obj.box)) for obj in review.objects}
    actual = {}
    for box in boxes:
        if not isinstance(box["xyxyPixels"], list) or any(
            type(n) not in (int, float) for n in box["xyxyPixels"]
        ):
            raise ValueError("coordinate receipt coordinates must be numbers")
        checked = ReviewObject.model_validate_json(
            json.dumps(
                {
                    "id": box["id"],
                    "class_id": box["classId"],
                    "class_name": box["className"],
                    "box": box["xyxyPixels"],
                }
            )
        )
        actual[checked.id] = (checked.class_id, checked.class_name, list(checked.box))
    if len(actual) != len(boxes) or actual != expected:
        raise ValueError("coordinate receipt coordinates differ")
