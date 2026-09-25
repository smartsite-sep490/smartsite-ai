import hashlib
import json
from pathlib import Path

from PIL import Image

from smartsite_ai.evaluation.dataset import load_evaluation_dataset
from smartsite_ai.inference.loading import load_artifact_spec
from smartsite_ai.tools.build_artifact_spec import run as run_artifact_spec
from smartsite_ai.tools.build_evaluation_dataset import run as run_evaluation_dataset
from smartsite_ai.training.dataset_integrity import aggregate_inventory, inventory_record

CANONICAL_CLASS_MAP = {
    "0": "Person",
    "1": "Hardhat",
    "2": "NO-Hardhat",
    "3": "Safety Vest",
    "4": "NO-Safety Vest",
}


def _prepared_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    colors = {"train": "red", "val": "green", "test": "blue"}
    for split, color in colors.items():
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image = images / f"{split}.jpg"
        Image.new("RGB", (100, 80), color=color).save(image, format="JPEG")
        (labels / f"{split}.txt").write_text(
            "0 0.5 0.5 0.8 0.9\n1 0.5 0.2 0.2 0.2\n", encoding="utf-8"
        )
    data = root / "data.yaml"
    data.write_text(
        "\n".join(
            (
                f"path: {json.dumps(root.resolve().as_posix())}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "nc: 5",
                "names: [Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest]",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output_files = [
        inventory_record(path, root)
        for path in sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    ]
    (root / "preparation.manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "status": "COMPLETE",
                "source": {
                    "url": "https://universe.roboflow.com/example/ppe/dataset/27",
                    "version": 27,
                    "license": "CC BY 4.0",
                },
                "canonical": {
                    "outputRoot": str(root.resolve()),
                    "classMap": CANONICAL_CLASS_MAP,
                },
                "outputFiles": output_files,
                "outputAggregate": {
                    "algorithm": "sha256-canonical-json-output-files-v1",
                    "sha256": aggregate_inventory(output_files),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return data


def _training_manifest(tmp_path: Path) -> tuple[Path, Path]:
    data = _prepared_dataset(tmp_path)
    preparation = json.loads((data.parent / "preparation.manifest.json").read_text())
    run = tmp_path / "runs" / "yolo11s-ppe-run-001"
    checkpoint = run / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fine-tuned yolo11s checkpoint")
    manifest = run / "training.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "status": "COMPLETE",
                "configuration": {
                    "name": "yolo11s-ppe-run-001",
                    "imageSize": 640,
                    "resolvedDevice": "cuda:0",
                    "dataConfig": str(data.resolve()),
                    "dataConfigSha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    "datasetAggregateSha256": preparation["outputAggregate"]["sha256"],
                },
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "sizeBytes": checkpoint.stat().st_size,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, checkpoint


def test_build_evaluation_dataset_publishes_loadable_atomic_output(tmp_path: Path) -> None:
    data = _prepared_dataset(tmp_path)
    output = tmp_path / "evaluation"

    assert (
        run_evaluation_dataset(
            [
                "--data",
                str(data.resolve()),
                "--output-dir",
                str(output.resolve()),
                "--dataset-id",
                "construction-site-safety",
                "--dataset-version",
                "27-smartsite-5class-v1",
            ]
        )
        == 0
    )

    loaded = load_evaluation_dataset(output / "evaluation.manifest.json")
    assert tuple(loaded.splits) == ("test", "train", "validation")
    assert {name: len(frames) for name, frames in loaded.splits.items()} == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    train = loaded.get_split("train")[0]
    assert train.media_path == "media/train/train.jpg"
    assert [item.class_name for item in train.annotations] == ["Person", "Hardhat"]
    conversion = json.loads((output / "conversion.manifest.json").read_text())
    assert conversion["status"] == "COMPLETE"
    assert conversion["evaluation"]["aggregateSha256"] == loaded.manifest.aggregate_sha256
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_build_evaluation_dataset_rejects_changed_prepared_byte_without_output(
    tmp_path: Path,
) -> None:
    data = _prepared_dataset(tmp_path)
    (data.parent / "train" / "labels" / "train.txt").write_text(
        "0 0.4 0.5 0.8 0.9\n", encoding="utf-8"
    )
    output = tmp_path / "evaluation"

    assert (
        run_evaluation_dataset(
            [
                "--data",
                str(data.resolve()),
                "--output-dir",
                str(output.resolve()),
                "--dataset-id",
                "construction-site-safety",
                "--dataset-version",
                "27-smartsite-5class-v1",
            ]
        )
        == 1
    )
    assert not output.exists()


def test_build_artifact_spec_uses_exact_complete_run_identity(tmp_path: Path) -> None:
    manifest, checkpoint = _training_manifest(tmp_path)
    output = tmp_path / "artifact.json"

    assert (
        run_artifact_spec(
            [
                "--training-manifest",
                str(manifest.resolve()),
                "--output",
                str(output.resolve()),
                "--artifact-id",
                "smartsite-yolo11s-ppe",
                "--source-url",
                "https://github.com/smartsite-sep490/model-releases/releases/download/v1/best.pt",
                "--license",
                "AGPL-3.0-only",
                "--license-reviewed",
                "--device",
                "trained",
            ]
        )
        == 0
    )

    spec = load_artifact_spec(output)
    assert spec.version == "yolo11s-ppe-run-001"
    assert spec.artifact_path == checkpoint.resolve()
    assert spec.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert dict(spec.class_map) == {int(key): value for key, value in CANONICAL_CLASS_MAP.items()}
    assert spec.device == "cuda:0"
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_build_artifact_spec_rejects_unreviewed_license_without_output(tmp_path: Path) -> None:
    manifest, _checkpoint = _training_manifest(tmp_path)
    output = tmp_path / "artifact.json"

    assert (
        run_artifact_spec(
            [
                "--training-manifest",
                str(manifest.resolve()),
                "--output",
                str(output.resolve()),
                "--source-url",
                "https://example.com/best.pt",
                "--license",
                "unverified",
                "--license-reviewed",
            ]
        )
        == 1
    )
    assert not output.exists()


def test_build_artifact_spec_rejects_non_public_source_url(tmp_path: Path) -> None:
    manifest, _checkpoint = _training_manifest(tmp_path)
    output = tmp_path / "artifact.json"

    assert (
        run_artifact_spec(
            [
                "--training-manifest",
                str(manifest.resolve()),
                "--output",
                str(output.resolve()),
                "--source-url",
                "https://127.0.0.1/best.pt",
                "--license",
                "AGPL-3.0-only",
                "--license-reviewed",
            ]
        )
        == 1
    )
    assert not output.exists()


def test_build_artifact_spec_rejects_checkpoint_hash_drift(tmp_path: Path) -> None:
    manifest, checkpoint = _training_manifest(tmp_path)
    checkpoint.write_bytes(b"mutated")
    output = tmp_path / "artifact.json"

    assert (
        run_artifact_spec(
            [
                "--training-manifest",
                str(manifest.resolve()),
                "--output",
                str(output.resolve()),
                "--source-url",
                "https://example.com/best.pt",
                "--license",
                "AGPL-3.0-only",
                "--license-reviewed",
            ]
        )
        == 1
    )
    assert not output.exists()
