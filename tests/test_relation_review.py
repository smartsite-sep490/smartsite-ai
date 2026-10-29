import copy
import hashlib
import importlib
import importlib.util
import json

import pytest
from PIL import Image
from pydantic import ValidationError


def module():
    name = "smartsite_ai.evaluation.relation_review"
    assert importlib.util.find_spec(name), "versioned relation intake is not implemented"
    return importlib.import_module(name)


def draft():
    return {
        "schema_version": "smartsite.ppe-relation-review/1",
        "status": "DRAFT_ONLY",
        "image_id": "sample",
        "source_sha256": "a" * 64,
        "coordinate_receipt_sha256": "b" * 64,
        "source_group": "proposed-scene-1",
        "encoded_dimensions": [100, 200],
        "exif_orientation": 1,
        "display_dimensions": [100, 200],
        "coordinate_space": "EXIF_NORMALIZED_DISPLAY_PIXELS",
        "objects": [
            {"id": "person", "class_id": 0, "class_name": "Person", "box": [5, 5, 90, 190]},
            {"id": "goggles", "class_id": 9, "class_name": "Goggles", "box": [20, 40, 60, 55]},
        ],
        "relations": [
            {
                "object_id": "goggles",
                "person_id": "person",
                "body_part": "EYES",
                "visibility": "UNRESOLVED",
                "proposal": "UNRESOLVED",
                "body_part_box": None,
            }
        ],
    }


def parse(value):
    return module().RelationReview.model_validate_json(json.dumps(value))


def test_draft_never_becomes_wearer_reference():
    value = parse(draft())
    assert value.status == "DRAFT_ONLY"
    assert value.relations[0].proposal == "UNRESOLVED"
    changed = draft()
    changed["model_approved"] = True
    with pytest.raises(ValidationError):
        parse(changed)


@pytest.mark.parametrize(
    "case",
    [
        "orphan",
        "wrong_owner",
        "duplicate",
        "missing_relation",
        "wrong_part",
        "negative",
        "name",
        "bounds",
        "bool_class",
        "approved",
        "wrong_dimensions",
        "nan",
    ],
)
def test_refuses_invalid_or_overclaimed_review(case):
    value = draft()
    relation = value["relations"][0]
    if case == "orphan":
        relation["object_id"] = "missing"
    elif case == "wrong_owner":
        relation["person_id"] = "goggles"
    elif case == "duplicate":
        value["relations"].append(copy.deepcopy(relation))
    elif case == "missing_relation":
        value["relations"] = []
    elif case == "wrong_part":
        relation["body_part"] = "FEET"
    elif case == "negative":
        value["objects"][1].update(class_id=10, class_name="NO-Goggles")
    elif case == "name":
        value["objects"][1]["class_name"] = "Boots"
    elif case == "bounds":
        value["objects"][1]["box"][2] = 101
    elif case == "bool_class":
        value["objects"][1]["class_id"] = True
    elif case == "approved":
        value["status"] = "APPROVED"
    elif case == "wrong_dimensions":
        value["exif_orientation"] = 6
    elif case == "nan":
        value["objects"][1]["box"][0] = float("nan")
    with pytest.raises(ValidationError):
        parse(value)


def test_loose_boots_can_have_no_person_without_becoming_missing():
    value = draft()
    value["objects"] = [
        {"id": "boot", "class_id": 7, "class_name": "Boots", "box": [5, 5, 70, 180]}
    ]
    value["relations"] = [
        {
            "object_id": "boot",
            "person_id": None,
            "body_part": "FEET",
            "visibility": "UNRESOLVED",
            "proposal": "UNRESOLVED",
            "body_part_box": None,
        }
    ]
    assert parse(value).relations[0].person_id is None


def test_worn_proposal_requires_visible_body_roi_but_is_still_draft():
    value = draft()
    relation = value["relations"][0]
    relation["proposal"] = "WORN_AT_EXPECTED_PART"
    with pytest.raises(ValidationError):
        parse(value)
    relation.update(visibility="VISIBLE", body_part_box=[18, 35, 65, 60])
    assert parse(value).status == "DRAFT_ONLY"
    relation["person_id"] = None
    with pytest.raises(ValidationError):
        parse(value)


def test_hanging_proposal_does_not_require_worn_visibility():
    value = draft()
    value["relations"][0]["proposal"] = "CARRIED_OR_HANGING"
    assert parse(value).relations[0].visibility == "UNRESOLVED"


def test_raw_only_families_cannot_be_wearer_relations():
    value = draft()
    value["objects"].append(
        {"id": "mask", "class_id": 11, "class_name": "Mask", "box": [15, 60, 70, 85]}
    )
    assert len(parse(value).objects) == 3
    value["relations"][0]["object_id"] = "mask"
    with pytest.raises(ValidationError):
        parse(value)


def test_source_binding_checks_hash_exif_and_dimensions(tmp_path):
    image_path = tmp_path / "source.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (200, 100), "white").save(image_path, exif=exif)
    value = draft()
    value.update(encoded_dimensions=[200, 100], exif_orientation=6)
    value["source_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    module().verify_source_image(parse(value), image_path)
    wrong = copy.deepcopy(value)
    wrong["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="hash"):
        module().verify_source_image(parse(wrong), image_path)
    wrong = copy.deepcopy(value)
    wrong.update(encoded_dimensions=[100, 200], exif_orientation=1)
    with pytest.raises(ValueError, match="orientation|dimensions"):
        module().verify_source_image(parse(wrong), image_path)


def test_loader_refuses_duplicate_keys_and_oversize(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text('{"status":"DRAFT_ONLY","status":"APPROVED"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module().load_relation_review(path)
    path.write_bytes(b" " * (module().MAX_REVIEW_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        module().load_relation_review(path)


def test_coordinate_receipt_must_bind_exact_objects_not_wearing(tmp_path):
    value = draft()
    image = {
        "id": value["image_id"],
        "sourceSha256": value["source_sha256"],
        "displayDimensions": value["display_dimensions"],
        "coordinateSpace": value["coordinate_space"],
        "boxes": [
            {
                "id": obj["id"],
                "classId": obj["class_id"],
                "className": obj["class_name"],
                "xyxyPixels": obj["box"],
            }
            for obj in value["objects"]
        ],
    }
    receipt = {
        "images": [image],
        "approvedPositiveBoxes": 2,
        "imageCount": 1,
        "answer": "synthetic coordinate approval",
        "questionItemId": "synthetic",
        "scope": "positive coordinates only",
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    value["coordinate_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    mod = module()
    assert hasattr(mod, "verify_coordinate_receipt"), "receipt hash is not verified yet"
    mod.verify_coordinate_receipt(parse(value), path)
    changed = copy.deepcopy(value)
    changed["objects"][1]["box"][0] += 1
    with pytest.raises(ValueError, match="coordinates"):
        mod.verify_coordinate_receipt(parse(changed), path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        mod.verify_coordinate_receipt(parse(value), path)


@pytest.mark.parametrize("receipt", [[], {"images": None}, {"images": [False]}])
def test_malformed_receipts_raise_validation_error(tmp_path, receipt):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    value = draft()
    value["coordinate_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module().verify_coordinate_receipt(parse(value), path)


@pytest.mark.parametrize("case", ["bool_class", "bool_coordinate", "orientation", "encoded"])
def test_receipt_rejects_coercible_values_and_conflicting_orientation(tmp_path, case):
    value = draft()
    value["objects"][0]["box"][0] = 0
    boxes = [
        {
            "id": obj["id"],
            "classId": obj["class_id"],
            "className": obj["class_name"],
            "xyxyPixels": obj["box"].copy(),
        }
        for obj in value["objects"]
    ]
    image = {
        "id": "sample",
        "sourceSha256": value["source_sha256"],
        "displayDimensions": [100, 200],
        "encodedDimensions": [100, 200],
        "exifOrientation": 1,
        "coordinateSpace": value["coordinate_space"],
        "boxes": boxes,
    }
    if case == "bool_class":
        boxes[0]["classId"] = False
    elif case == "bool_coordinate":
        boxes[0]["xyxyPixels"][0] = False
    elif case == "orientation":
        image["exifOrientation"] = 6
    elif case == "encoded":
        image["encodedDimensions"] = [200, 100]
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"images": [image], "questionItemId": "synthetic", "answer": "yes"}))
    value["coordinate_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module().verify_coordinate_receipt(parse(value), path)
