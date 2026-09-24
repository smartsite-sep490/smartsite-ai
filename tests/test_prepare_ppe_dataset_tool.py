import json
from pathlib import Path

import pytest

from smartsite_ai.tools.prepare_ppe_dataset import (
    CANONICAL_CLASS_MAP,
    SOURCE_CLASS_MAP,
    DatasetPreparationError,
    prepare_dataset,
    run,
)


def _source(
    tmp_path: Path, *, label: str = "5 0.5 0.5 0.4 0.8\n0 0.5 0.2 0.2 0.2\n1 0.1 0.1 0.1 0.1\n"
) -> Path:
    root = tmp_path / "source"
    names = ", ".join(name for _, name in SOURCE_CLASS_MAP)
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "\n".join(
            (
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                "nc: 10",
                f"names: [{names}]",
                "roboflow:",
                "  license: CC BY 4.0",
                "  project: construction-site-safety",
                "  url: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27",
                "  version: 27",
                "  workspace: roboflow-universe-projects",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{split}.jpg").write_bytes(f"image-{split}".encode())
        (labels / f"{split}.txt").write_text(label, encoding="utf-8")
    return root


def test_prepare_remaps_labels_and_writes_complete_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared"

    manifest_path = prepare_dataset(source.resolve(), output.resolve())

    assert manifest_path == output / "preparation.manifest.json"
    assert (output / "train" / "labels" / "train.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.4 0.8\n1 0.5 0.2 0.2 0.2\n"
    )
    assert (output / "data.yaml").read_text(encoding="utf-8").endswith("  4: NO-Safety Vest\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["source"]["version"] == 27
    assert manifest["source"]["license"] == "CC BY 4.0"
    assert manifest["canonical"]["classMap"] == {
        str(class_id): name for class_id, name in CANONICAL_CLASS_MAP
    }
    assert manifest["counts"]["retainedByClass"] == {"Hardhat": 3, "Person": 3}
    assert manifest["counts"]["droppedByClass"] == {"Mask": 3}
    assert len(manifest["inputFiles"]) == 7
    assert len(manifest["outputFiles"]) == 7
    assert len(manifest["outputAggregate"]["sha256"]) == 64


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("10 0.5 0.5 0.2 0.2\n", "unknown class ID"),
        ("5 nan 0.5 0.2 0.2\n", "finite and normalized"),
        ("5 0.5 0.5 0 0.2\n", "width and height must be positive"),
        ("5 0.5 0.5 0.2\n", "malformed YOLO row"),
    ],
)
def test_prepare_rejects_invalid_labels_without_publishing_output(
    tmp_path: Path, label: str, message: str
) -> None:
    source = _source(tmp_path, label=label)
    output = tmp_path / "prepared"

    with pytest.raises(DatasetPreparationError, match=message):
        prepare_dataset(source.resolve(), output.resolve())

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.*"))


def test_prepare_requires_one_to_one_image_label_pairs(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "valid" / "labels" / "valid.txt").unlink()

    with pytest.raises(DatasetPreparationError, match="no paired label"):
        prepare_dataset(source.resolve(), (tmp_path / "prepared").resolve())


def test_prepare_rejects_unreviewed_source_class_map(tmp_path: Path) -> None:
    source = _source(tmp_path)
    config = source / "data.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("Hardhat, Mask", "Helmet, Mask"),
        encoding="utf-8",
    )

    with pytest.raises(DatasetPreparationError, match="class map must exactly match"):
        prepare_dataset(source.resolve(), (tmp_path / "prepared").resolve())


def test_cli_rejects_existing_output_without_modifying_it(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert run(["--input-dir", str(source.resolve()), "--output-dir", str(output.resolve())]) == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
