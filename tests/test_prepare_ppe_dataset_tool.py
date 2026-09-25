import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import smartsite_ai.tools.prepare_ppe_dataset as preparation
from smartsite_ai.tools.prepare_ppe_dataset import (
    CANONICAL_CLASS_MAP,
    SOURCE_CLASS_MAP,
    DatasetPreparationError,
    _require_ignored_repository_path,
    _validate_image,
    inspect_source,
    prepare_dataset,
    run,
)


@pytest.fixture(autouse=True)
def _small_reviewed_split_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparation, "EXPECTED_SPLIT_COUNTS", {"train": 1, "val": 1, "test": 1})


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
        Image.new("RGB", (16, 16), color=(12, 34, 56)).save(images / f"{split}.jpg")
        (labels / f"{split}.txt").write_text(label, encoding="utf-8")
    return root


def test_prepare_remaps_labels_and_writes_complete_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared"

    aggregate = inspect_source(source.resolve()).aggregate_sha256
    manifest_path = prepare_dataset(
        source.resolve(), output.resolve(), expected_source_aggregate=aggregate
    )

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
        prepare_dataset(
            source.resolve(),
            output.resolve(),
            expected_source_aggregate=inspect_source(source.resolve()).aggregate_sha256,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.*"))


def test_prepare_requires_one_to_one_image_label_pairs(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "valid" / "labels" / "valid.txt").unlink()

    with pytest.raises(DatasetPreparationError, match="no paired label"):
        prepare_dataset(
            source.resolve(),
            (tmp_path / "prepared").resolve(),
            expected_source_aggregate=inspect_source(source.resolve()).aggregate_sha256,
        )


def test_inspection_rejects_unreviewed_extra_pair(tmp_path: Path) -> None:
    source = _source(tmp_path)
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(source / "train" / "images" / "extra.jpg")
    (source / "train" / "labels" / "extra.txt").write_text("5 0.5 0.5 0.4 0.8\n", encoding="utf-8")

    with pytest.raises(
        DatasetPreparationError, match="train split must contain the reviewed 1 image-label pairs"
    ):
        inspect_source(source.resolve())


def test_prepare_rejects_unreviewed_source_class_map(tmp_path: Path) -> None:
    source = _source(tmp_path)
    config = source / "data.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("Hardhat, Mask", "Helmet, Mask"),
        encoding="utf-8",
    )

    with pytest.raises(DatasetPreparationError, match="class map must exactly match"):
        prepare_dataset(
            source.resolve(),
            (tmp_path / "prepared").resolve(),
            expected_source_aggregate="0" * 64,
        )


def test_cli_rejects_existing_output_without_modifying_it(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert (
        run(
            [
                "--input-dir",
                str(source.resolve()),
                "--output-dir",
                str(output.resolve()),
                "--expected-source-aggregate",
                inspect_source(source.resolve()).aggregate_sha256,
            ]
        )
        == 1
    )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_prepare_rejects_corrupt_image_without_publishing(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "test" / "images" / "test.jpg").write_bytes(b"not-an-image")

    with pytest.raises(DatasetPreparationError, match="corrupt or unsupported"):
        inspect_source(source.resolve())


def test_image_bounds_are_checked_before_pixel_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "oversized.jpg"
    image_path.write_bytes(b"header")
    load_called = False

    class OversizedImage:
        size = (16_385, 1)
        format = "JPEG"

        def __enter__(self) -> "OversizedImage":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify(self) -> None:
            return None

        def load(self) -> None:
            nonlocal load_called
            load_called = True

    monkeypatch.setattr(Image, "open", lambda _path: OversizedImage())

    with pytest.raises(DatasetPreparationError, match="dimensions"):
        _validate_image(image_path)

    assert load_called is False


def test_prepare_rejects_linked_directory_before_traversal(tmp_path: Path) -> None:
    source = _source(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "outside.jpg").write_bytes(b"outside")
    link = source / "train" / "images" / "escaped"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")

    with pytest.raises(DatasetPreparationError, match="symlink/junction/reparse"):
        inspect_source(source.resolve())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_prepare_rejects_windows_junction_before_traversal(tmp_path: Path) -> None:
    source = _source(tmp_path)
    external = tmp_path / "junction-target"
    external.mkdir()
    (external / "outside.jpg").write_bytes(b"outside")
    junction = source / "train" / "images" / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("Windows junction creation is unavailable on this host")
    try:
        with pytest.raises(DatasetPreparationError, match="symlink/junction/reparse"):
            inspect_source(source.resolve())
    finally:
        junction.rmdir()


def test_prepare_requires_pinned_inspected_source_bytes(tmp_path: Path) -> None:
    source = _source(tmp_path)

    with pytest.raises(DatasetPreparationError, match="does not match"):
        prepare_dataset(
            source.resolve(),
            (tmp_path / "prepared").resolve(),
            expected_source_aggregate="0" * 64,
        )


def test_inspect_only_prints_aggregate_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(tmp_path)

    assert run(["--input-dir", str(source.resolve()), "--inspect-only"]) == 0

    assert inspect_source(source.resolve()).aggregate_sha256 in capsys.readouterr().out
    assert not (tmp_path / "prepared").exists()


def test_git_ignore_check_uses_output_repository_not_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "--quiet"], check=True)
    (repository / ".gitignore").write_text("local-data/\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    _require_ignored_repository_path(repository / "local-data" / "prepared")
    with pytest.raises(DatasetPreparationError, match="must be ignored"):
        _require_ignored_repository_path(repository / "tracked-output")
