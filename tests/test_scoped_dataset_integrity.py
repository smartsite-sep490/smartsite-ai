"""Synthetic admission checks; no real images, weights or external services."""

import json
from pathlib import Path

import pytest

from smartsite_ai.inference.ppe_profiles import UNIFIED_PPE_15_PROFILE
from smartsite_ai.training.dataset_integrity import (
    DatasetIntegrityError,
    aggregate_inventory,
    inventory_record,
    sha256_file,
    verify_prepared_dataset,
)


def _fixture(root: Path) -> tuple[Path, list[dict[str, object]]]:
    rows = []
    for split in ("train", "val"):
        image = root / split / "images" / "sample.jpg"
        label = root / split / "labels" / "sample.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(split.encode())
        label.write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append(
            {
                "path": image.relative_to(root).as_posix(),
                "imageSha256": sha256_file(image),
                "scope": [index == 9 for index in range(15)],
                "positiveOnlyClassId": 9,
                "sourceId": "synthetic-approved-positive",
                "labelOrigin": "synthetic-test",
            }
        )
    data = root / "data.yaml"
    data.write_text("# synthetic inventory fixture\n", encoding="utf-8")
    _seal(root, rows)
    return data, rows


def _seal(root: Path, rows: list[dict[str, object]], **overrides: object) -> None:
    (root / "supervision.scopes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    files = [
        inventory_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "preparation.manifest.json"
    ]
    manifest = {
        "schemaVersion": "1.0.0",
        "status": "COMPLETE",
        "experimentalOnly": True,
        "license": "synthetic fixtures only",
        "attribution": "SmartSite tests",
        "canonical": {
            "outputRoot": str(root.resolve()),
            "classMap": {str(key): name for key, name in UNIFIED_PPE_15_PROFILE.class_map},
        },
        "outputFiles": files,
        "outputAggregate": {
            "algorithm": "sha256-canonical-json-output-files-v1",
            "sha256": aggregate_inventory(files),
        },
        **overrides,
    }
    (root / "preparation.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_experimental_admission_requires_explicit_profile_and_preserves_legacy(tmp_path: Path):
    data, _ = _fixture(tmp_path)
    assert len(verify_prepared_dataset(data, expected_profile="unified-ppe-15")) == 64
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "extra",
        "stale",
        "numeric_scope",
        "empty_scope",
        "wrong_width",
        "bool_mode",
        "wrong_mode",
        "extra_scope",
        "outside_positive",
        "source_missing",
        "origin_missing",
        "unknown_field",
        "noncanonical_path",
        "bad_label",
        "missing_label",
    ],
)
def test_scopes_fail_closed_even_when_inventory_is_resealed(tmp_path: Path, change: str):
    data, rows = _fixture(tmp_path)
    row = rows[0]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(row))
    elif change == "extra":
        row["path"] = "train/images/unknown.jpg"
    elif change == "stale":
        row["imageSha256"] = "a" * 64
    elif change == "numeric_scope":
        row["scope"] = [int(index == 9) for index in range(15)]
    elif change == "empty_scope":
        row["scope"] = [False] * 15
    elif change == "wrong_width":
        row["scope"] = [True]
    elif change == "bool_mode":
        row["positiveOnlyClassId"] = True
    elif change == "wrong_mode":
        row["positiveOnlyClassId"] = 15
    elif change == "extra_scope":
        row["scope"] = [True] * 15
    elif change == "outside_positive":
        (tmp_path / "train/labels/sample.txt").write_text("5 0.5 0.5 0.2 0.2\n")
    elif change == "source_missing":
        row["sourceId"] = " "
    elif change == "origin_missing":
        row["labelOrigin"] = ""
    elif change == "unknown_field":
        row["approved"] = True
    elif change == "noncanonical_path":
        row["path"] = "train/images/../images/sample.jpg"
    elif change == "bad_label":
        (tmp_path / "train/labels/sample.txt").write_text("9 nan 0.5 0.2 0.2\n")
    elif change == "missing_label":
        (tmp_path / "train/labels/sample.txt").unlink()
    _seal(tmp_path, rows)
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile="unified-ppe-15")


@pytest.mark.parametrize(
    "field,value", [("experimentalOnly", False), ("license", ""), ("attribution", " ")]
)
def test_experimental_metadata_is_required(tmp_path: Path, field: str, value: object):
    data, rows = _fixture(tmp_path)
    _seal(tmp_path, rows, **{field: value})
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile="unified-ppe-15")


def test_complete_class_scope_and_empty_labels_are_valid(tmp_path: Path):
    data, rows = _fixture(tmp_path)
    for row in rows:
        row["positiveOnlyClassId"] = -1
    (tmp_path / "val/labels/sample.txt").write_text("")
    _seal(tmp_path, rows)
    assert verify_prepared_dataset(data, expected_profile="unified-ppe-15")


@pytest.mark.parametrize(
    "change",
    ["nested_image", "extra_split_image", "root_images", "orphan_label", "overlap", "no_scopes"],
)
def test_no_unscoped_or_cross_split_data_is_admitted(tmp_path: Path, change: str):
    data, rows = _fixture(tmp_path)
    if change in {"nested_image", "extra_split_image", "root_images"}:
        relative = {
            "nested_image": "train/images/nested/extra.jpg",
            "extra_split_image": "extra/images/a.jpg",
            "root_images": "images/a.jpg",
        }[change]
        image = tmp_path / relative
        image.parent.mkdir(parents=True)
        image.write_bytes(b"unreviewed image")
    elif change == "orphan_label":
        (tmp_path / "train/labels/orphan.txt").write_text("")
    elif change == "overlap":
        (tmp_path / "val/images/sample.jpg").write_bytes(b"train")
        rows[1]["imageSha256"] = rows[0]["imageSha256"]
    _seal(tmp_path, rows)
    if change == "no_scopes":
        (tmp_path / "supervision.scopes.jsonl").unlink()
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile="unified-ppe-15")


@pytest.mark.parametrize("change", ["empty_positive", "wrong_single_bit", "stem_collision"])
def test_positive_mode_and_image_identity_boundaries(tmp_path: Path, change: str):
    data, rows = _fixture(tmp_path)
    if change == "empty_positive":
        (tmp_path / "train/labels/sample.txt").write_text("")
    elif change == "wrong_single_bit":
        rows[0]["scope"] = [index == 8 for index in range(15)]
    else:
        (tmp_path / "train/images/sample.png").write_bytes(b"second image same label name")
    _seal(tmp_path, rows)
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile="unified-ppe-15")


@pytest.mark.parametrize("profile", ["legacy-ppe-5", "native-ppe-10", "unknown"])
def test_other_profiles_cannot_admit_unified_data(tmp_path: Path, profile: str):
    data, _ = _fixture(tmp_path)
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile=profile)


@pytest.mark.parametrize("change", ["json", "trailing", "tamper", "test_missing_scope"])
def test_corrupted_or_unscoped_additions_fail(tmp_path: Path, change: str):
    data, rows = _fixture(tmp_path)
    if change == "test_missing_scope":
        for kind, contents in (("images", b"test image"), ("labels", b"")):
            file = tmp_path / "test" / kind / ("test.jpg" if kind == "images" else "test.txt")
            file.parent.mkdir(parents=True)
            file.write_bytes(contents)
        _seal(tmp_path, rows)
    else:
        scope_path = tmp_path / "supervision.scopes.jsonl"
        with scope_path.open("a") as handle:
            handle.write("bad json\n" if change == "json" else "{}\n")
        if change != "tamper":
            manifest = json.loads((tmp_path / "preparation.manifest.json").read_text())
            manifest["outputFiles"] = [
                inventory_record(tmp_path / item["path"], tmp_path)
                for item in manifest["outputFiles"]
            ]
            manifest["outputAggregate"]["sha256"] = aggregate_inventory(manifest["outputFiles"])
            (tmp_path / "preparation.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetIntegrityError):
        verify_prepared_dataset(data, expected_profile="unified-ppe-15")


def test_complete_multiclass_scope_can_include_optional_test_split(tmp_path: Path):
    data, rows = _fixture(tmp_path)
    for row in rows:
        row["scope"] = [index < 11 for index in range(15)]
        row["positiveOnlyClassId"] = -1
    image = tmp_path / "test/images/test.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"distinct test")
    label = tmp_path / "test/labels/test.txt"
    label.parent.mkdir(parents=True)
    label.write_text("5 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n")
    rows.append({**rows[0], "path": "test/images/test.jpg", "imageSha256": sha256_file(image)})
    _seal(tmp_path, rows)
    assert verify_prepared_dataset(data, expected_profile="unified-ppe-15")
