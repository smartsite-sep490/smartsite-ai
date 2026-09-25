"""Build a strict local evaluation dataset from a verified prepared YOLO dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid5

from PIL import Image
from pydantic import ValidationError

from smartsite_ai.evaluation.dataset import (
    compute_dataset_aggregate_sha256,
    load_evaluation_dataset,
)
from smartsite_ai.evaluation.models import EvaluationFrame, EvaluationManifest
from smartsite_ai.inference.loading import CANONICAL_PPE_CLASS_MAP
from smartsite_ai.training.dataset_integrity import (
    DatasetIntegrityError,
    is_link_like,
    safe_contained_path,
    sha256_file,
    verify_prepared_dataset,
    walk_safe_files,
)

_MANIFEST_FILENAME = "evaluation.manifest.json"
_CONVERSION_MANIFEST_FILENAME = "conversion.manifest.json"
_SPLITS: tuple[tuple[str, str], ...] = (
    ("train", "train"),
    ("validation", "val"),
    ("test", "test"),
)
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_MAX_LABEL_BYTES = 4 * 1024 * 1024
_MAX_ANNOTATIONS_PER_IMAGE = 500
_CLASS_MAP = {str(class_id): name for class_id, name in CANONICAL_PPE_CLASS_MAP}


class EvaluationDatasetBuildError(ValueError):
    """Raised when verified training data cannot become an evaluation dataset."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsite-ai-build-evaluation-dataset",
        description=(
            "Convert one verified canonical five-class YOLO dataset into the local "
            "SmartSite evaluation manifest and split indexes."
        ),
    )
    parser.add_argument("--data", required=True, type=Path, help="absolute prepared data.yaml")
    parser.add_argument("--output-dir", required=True, type=Path, help="absolute new output dir")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationDatasetBuildError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationDatasetBuildError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise EvaluationDatasetBuildError(f"{label} root must be an object")
    return value


def _absolute_regular_file(path: Path, *, label: str, suffix: str) -> Path:
    if not path.is_absolute():
        raise EvaluationDatasetBuildError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationDatasetBuildError(f"{label} must exist") from error
    if is_link_like(path) or not resolved.is_file() or resolved.suffix.lower() != suffix:
        raise EvaluationDatasetBuildError(f"{label} must be a non-link regular {suffix} file")
    return resolved


def _require_ignored_repository_path(path: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if result.returncode != 0:
        return
    repository_root = Path(result.stdout.strip()).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", "--no-index", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if ignored.returncode != 0:
        raise EvaluationDatasetBuildError("in-repository output directory must be ignored by Git")


def _new_output_directory(path: Path, *, source_root: Path) -> Path:
    if not path.is_absolute():
        raise EvaluationDatasetBuildError("output directory must be an absolute path")
    if path.exists() or path.is_symlink():
        raise EvaluationDatasetBuildError("output directory must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationDatasetBuildError("output parent must exist") from error
    output = parent / path.name
    if is_link_like(parent) or not parent.is_dir():
        raise EvaluationDatasetBuildError("output parent must be a non-link directory")
    if output == source_root or source_root in output.parents:
        raise EvaluationDatasetBuildError(
            "output directory must not be inside the prepared dataset"
        )
    _require_ignored_repository_path(output)
    return output


def _load_data_yaml(path: Path) -> Mapping[str, object]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError as error:
        raise EvaluationDatasetBuildError("PyYAML is unavailable") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvaluationDatasetBuildError("data config must be readable UTF-8 YAML") from error
    if not isinstance(value, Mapping):
        raise EvaluationDatasetBuildError("data config root must be a mapping")
    return value


def _normalize_class_map(value: object) -> dict[str, str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return {str(index): item for index, item in enumerate(value)}
    if not isinstance(value, Mapping):
        raise EvaluationDatasetBuildError("data config names must be a list or mapping")
    normalized: dict[str, str] = {}
    for key, name in value.items():
        if isinstance(key, bool) or not isinstance(key, int | str) or not isinstance(name, str):
            raise EvaluationDatasetBuildError("data config class map is invalid")
        try:
            class_id = int(key)
        except ValueError as error:
            raise EvaluationDatasetBuildError("data config class IDs must be integers") from error
        if str(class_id) != str(key):
            raise EvaluationDatasetBuildError("data config class IDs must be canonical integers")
        normalized[str(class_id)] = name
    return normalized


def _prepared_metadata(root: Path, data: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    if _normalize_class_map(data.get("names")) != _CLASS_MAP or data.get("nc") not in (None, 5):
        raise EvaluationDatasetBuildError("data config must use the exact canonical PPE class map")
    declared_root = data.get("path")
    if not isinstance(declared_root, str) or not Path(declared_root).is_absolute():
        raise EvaluationDatasetBuildError("data config path must be the absolute prepared root")
    try:
        if Path(declared_root).resolve(strict=True) != root:
            raise EvaluationDatasetBuildError("data config path does not match its prepared root")
    except (OSError, RuntimeError, ValueError) as error:
        raise EvaluationDatasetBuildError("data config path does not exist") from error
    try:
        aggregate = verify_prepared_dataset(root / "data.yaml")
    except DatasetIntegrityError as error:
        raise EvaluationDatasetBuildError(f"prepared dataset integrity failed: {error}") from error
    preparation = _load_json_object(
        root / "preparation.manifest.json", label="preparation manifest"
    )
    return aggregate, preparation


def _split_directories(
    root: Path, data: Mapping[str, object], canonical_name: str, provider_name: str
) -> tuple[Path, Path]:
    configured = data.get(provider_name)
    expected = f"{provider_name}/images"
    if configured != expected:
        raise EvaluationDatasetBuildError(
            f"data config {provider_name} must equal {expected!r} for canonical conversion"
        )
    images = root / provider_name / "images"
    labels = root / provider_name / "labels"
    if not images.is_dir() or not labels.is_dir() or is_link_like(images) or is_link_like(labels):
        raise EvaluationDatasetBuildError(f"{canonical_name} image/label directories are unsafe")
    return images, labels


def _bounded_box(fields: list[str], *, label_path: Path, line_number: int) -> dict[str, float]:
    try:
        center_x, center_y, width, height = (float(value) for value in fields)
    except ValueError as error:
        raise EvaluationDatasetBuildError(
            f"non-numeric YOLO coordinate at {label_path}:{line_number}"
        ) from error
    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) for value in values) or not all(
        0.0 <= value <= 1.0 for value in values
    ):
        raise EvaluationDatasetBuildError(
            f"YOLO coordinates must be finite and normalized at {label_path}:{line_number}"
        )
    x1, y1 = center_x - width / 2.0, center_y - height / 2.0
    x2, y2 = center_x + width / 2.0, center_y + height / 2.0
    if width <= 0.0 or height <= 0.0 or min(x1, y1) < 0.0 or max(x2, y2) > 1.0:
        raise EvaluationDatasetBuildError(
            f"YOLO box must remain inside the image at {label_path}:{line_number}"
        )
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _annotations(label_path: Path, frame_id: str, *, source_root: Path) -> list[dict[str, object]]:
    try:
        safe_contained_path(label_path, source_root, kind="paired label")
    except DatasetIntegrityError as error:
        raise EvaluationDatasetBuildError(str(error)) from error
    if is_link_like(label_path) or not label_path.is_file():
        raise EvaluationDatasetBuildError(f"paired label is missing or linked: {label_path}")
    if label_path.stat().st_size > _MAX_LABEL_BYTES:
        raise EvaluationDatasetBuildError(f"label exceeds size limit: {label_path}")
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationDatasetBuildError(f"label must be readable UTF-8: {label_path}") from error
    annotations: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in _CLASS_MAP:
            raise EvaluationDatasetBuildError(
                f"invalid canonical YOLO row at {label_path}:{line_number}"
            )
        annotations.append(
            {
                "annotationId": f"{frame_id}-a{len(annotations):04d}",
                "className": _CLASS_MAP[fields[0]],
                "boundingBox": _bounded_box(
                    fields[1:], label_path=label_path, line_number=line_number
                ),
                "visibility": 1.0,
            }
        )
        if len(annotations) > _MAX_ANNOTATIONS_PER_IMAGE:
            raise EvaluationDatasetBuildError(f"too many annotations in label: {label_path}")
    return annotations


def _image_paths(images: Path, *, source_root: Path) -> list[Path]:
    try:
        paths = sorted(walk_safe_files(images, source_root))
    except DatasetIntegrityError as error:
        raise EvaluationDatasetBuildError(str(error)) from error
    if not paths:
        raise EvaluationDatasetBuildError(f"image directory is empty: {images}")
    for path in paths:
        if is_link_like(path) or path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise EvaluationDatasetBuildError(f"unsupported or linked image: {path}")
    return paths


def _frame(
    *,
    image: Path,
    images_root: Path,
    labels_root: Path,
    split: str,
    prepared_aggregate: str,
    source_root: Path,
    temporary_root: Path,
) -> EvaluationFrame:
    relative = image.relative_to(images_root)
    label = labels_root / relative.with_suffix(".txt")
    frame_id = str(
        uuid5(
            NAMESPACE_URL,
            f"smartsite-ppe-evaluation:{prepared_aggregate}:{split}:{relative.as_posix()}",
        )
    )
    try:
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
    except Exception as error:
        raise EvaluationDatasetBuildError(f"image is corrupt or unsupported: {image}") from error
    media_relative = PurePosixPath("media", split, *relative.parts).as_posix()
    destination = temporary_root.joinpath(*PurePosixPath(media_relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, destination)
    source_hash = sha256_file(image)
    if sha256_file(destination) != source_hash:
        raise EvaluationDatasetBuildError(f"copied image checksum mismatch: {image}")
    try:
        return EvaluationFrame.model_validate(
            {
                "frameId": frame_id,
                "mediaPath": media_relative,
                "sha256": source_hash,
                "width": width,
                "height": height,
                "annotations": _annotations(label, frame_id, source_root=source_root),
            }
        )
    except ValidationError as error:
        raise EvaluationDatasetBuildError(
            f"invalid evaluation frame from {image}: {error}"
        ) from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, frames: Sequence[EvaluationFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(frame.model_dump_json(by_alias=True, exclude_none=True) + "\n")


def build_evaluation_dataset(
    data_config: Path,
    output_root: Path,
    *,
    dataset_id: str,
    dataset_version: str,
) -> Path:
    """Verify source bytes and atomically publish a loader-validated evaluation dataset."""

    data_config = _absolute_regular_file(data_config, label="data config", suffix=".yaml")
    source_root = data_config.parent
    data = _load_data_yaml(data_config)
    prepared_aggregate, preparation = _prepared_metadata(source_root, data)
    output_root = _new_output_directory(output_root, source_root=source_root)
    source = preparation.get("source")
    if not isinstance(source, Mapping):
        raise EvaluationDatasetBuildError("preparation manifest source metadata is missing")
    source_url, source_license = source.get("url"), source.get("license")
    if not isinstance(source_url, str) or not isinstance(source_license, str):
        raise EvaluationDatasetBuildError("preparation manifest source URL/license is invalid")
    try:
        EvaluationManifest.model_validate(
            {
                "schemaVersion": "1.0.0",
                "datasetId": dataset_id,
                "datasetVersion": dataset_version,
                "sourceUrl": source_url,
                "license": source_license,
                "aggregateSha256": "0" * 64,
                "classMap": _CLASS_MAP,
                "splits": {split: f"indexes/{split}.jsonl" for split, _provider in _SPLITS},
            }
        )
    except ValidationError as error:
        raise EvaluationDatasetBuildError(
            f"invalid evaluation manifest metadata: {error}"
        ) from error

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        split_frames: dict[str, tuple[EvaluationFrame, ...]] = {}
        for canonical_name, provider_name in _SPLITS:
            images, labels = _split_directories(source_root, data, canonical_name, provider_name)
            frames = tuple(
                _frame(
                    image=image,
                    images_root=images,
                    labels_root=labels,
                    split=canonical_name,
                    prepared_aggregate=prepared_aggregate,
                    source_root=source_root,
                    temporary_root=temporary,
                )
                for image in _image_paths(images, source_root=source_root)
            )
            split_frames[canonical_name] = frames
            _write_jsonl(temporary / "indexes" / f"{canonical_name}.jsonl", frames)

        aggregate = compute_dataset_aggregate_sha256(split_frames)
        try:
            manifest = EvaluationManifest.model_validate(
                {
                    "schemaVersion": "1.0.0",
                    "datasetId": dataset_id,
                    "datasetVersion": dataset_version,
                    "sourceUrl": source_url,
                    "license": source_license,
                    "aggregateSha256": aggregate,
                    "classMap": _CLASS_MAP,
                    "splits": {split: f"indexes/{split}.jsonl" for split, _provider in _SPLITS},
                }
            )
        except ValidationError as error:
            raise EvaluationDatasetBuildError(
                f"invalid evaluation manifest metadata: {error}"
            ) from error
        _write_json(temporary / _MANIFEST_FILENAME, manifest.model_dump(by_alias=True))
        _write_json(
            temporary / _CONVERSION_MANIFEST_FILENAME,
            {
                "schemaVersion": "1.0.0",
                "status": "COMPLETE",
                "source": {
                    "dataConfig": str(data_config),
                    "dataConfigSha256": sha256_file(data_config),
                    "preparationManifestSha256": sha256_file(
                        source_root / "preparation.manifest.json"
                    ),
                    "preparedDatasetAggregateSha256": prepared_aggregate,
                    "sourceUrl": source_url,
                    "license": source_license,
                },
                "evaluation": {
                    "manifest": _MANIFEST_FILENAME,
                    "aggregateSha256": aggregate,
                    "classMap": _CLASS_MAP,
                    "splitCounts": {
                        split: len(frames) for split, frames in sorted(split_frames.items())
                    },
                },
            },
        )
        loaded = load_evaluation_dataset(temporary / _MANIFEST_FILENAME)
        if loaded.manifest.aggregate_sha256 != aggregate:
            raise EvaluationDatasetBuildError("published evaluation aggregate is inconsistent")
        if verify_prepared_dataset(data_config) != prepared_aggregate:
            raise EvaluationDatasetBuildError("prepared dataset changed during conversion")
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root / _MANIFEST_FILENAME


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        manifest = build_evaluation_dataset(
            parsed.data,
            parsed.output_dir,
            dataset_id=parsed.dataset_id,
            dataset_version=parsed.dataset_version,
        )
    except KeyboardInterrupt:
        print("evaluation dataset conversion interrupted; no output was published", file=sys.stderr)
        return 130
    except (EvaluationDatasetBuildError, DatasetIntegrityError, OSError, RuntimeError) as error:
        print(f"evaluation dataset conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"evaluation dataset complete: {manifest}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
