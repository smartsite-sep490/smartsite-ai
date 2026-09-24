"""Deterministically prepare the reviewed Roboflow PPE dataset for SmartSite.

This module deliberately has no downloader. It accepts one existing local Roboflow
Construction Site Safety v27 YOLO export and publishes a separate five-class dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SOURCE_URL = (
    "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27"
)
SOURCE_VERSION = 27
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_ATTRIBUTION = "Roboflow Universe Projects — Construction Site Safety"
SOURCE_CLASS_MAP: tuple[tuple[int, str], ...] = (
    (0, "Hardhat"),
    (1, "Mask"),
    (2, "NO-Hardhat"),
    (3, "NO-Mask"),
    (4, "NO-Safety Vest"),
    (5, "Person"),
    (6, "Safety Cone"),
    (7, "Safety Vest"),
    (8, "machinery"),
    (9, "vehicle"),
)
CANONICAL_CLASS_MAP: tuple[tuple[int, str], ...] = (
    (0, "Person"),
    (1, "Hardhat"),
    (2, "NO-Hardhat"),
    (3, "Safety Vest"),
    (4, "NO-Safety Vest"),
)
SOURCE_TO_CANONICAL: Mapping[int, int] = {5: 0, 0: 1, 2: 2, 7: 3, 4: 4}
DROPPED_SOURCE_IDS = frozenset({1, 3, 6, 8, 9})
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_CLASS_ID_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)$")
_MANIFEST_FILENAME = "preparation.manifest.json"


class DatasetPreparationError(ValueError):
    """Raised when source data or a destination boundary is invalid."""


@dataclass(frozen=True, slots=True)
class PreparedFile:
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str
    input_size_bytes: int
    output_size_bytes: int


@dataclass(frozen=True, slots=True)
class SplitPlan:
    canonical_name: str
    source_name: str
    pairs: tuple[tuple[Path, Path, PurePosixPath], ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the local-only CLI without importing vision or network packages."""

    parser = argparse.ArgumentParser(
        prog="smartsite-ai-prepare-ppe-dataset",
        description=(
            "Filter and remap one local Roboflow Construction Site Safety v27 YOLO export "
            "to the canonical SmartSite five-class PPE dataset."
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="absolute source export dir")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="absolute new ignored destination dir"
    )
    return parser


def _absolute_existing_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise DatasetPreparationError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetPreparationError(f"{label} must be an existing directory") from error
    if path.is_symlink() or not resolved.is_dir():
        raise DatasetPreparationError(f"{label} must be a non-symlink directory")
    return resolved


def _validated_output(path: Path, input_root: Path) -> Path:
    if not path.is_absolute():
        raise DatasetPreparationError("output directory must be an absolute path")
    if path.exists() or path.is_symlink():
        raise DatasetPreparationError("output directory must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
        output = parent / path.name
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetPreparationError("output parent must be an existing directory") from error
    if not parent.is_dir() or parent.is_symlink():
        raise DatasetPreparationError("output parent must be a non-symlink directory")
    if output == input_root or input_root in output.parents:
        raise DatasetPreparationError("output directory must not be inside the source dataset")
    _require_ignored_repository_path(output)
    return output


def _require_ignored_repository_path(path: Path) -> None:
    """Reject generated output inside this repository unless Git ignores it."""

    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if root_result.returncode != 0:
        return
    repository_root = Path(root_result.stdout.strip()).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if ignored.returncode != 0:
        raise DatasetPreparationError("in-repository output directory must be ignored by Git")


def _normalized_class_map(value: object) -> tuple[tuple[int, str], ...]:
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise DatasetPreparationError("source data.yaml names must contain only strings")
        return tuple(enumerate(value))
    if not isinstance(value, Mapping):
        raise DatasetPreparationError("source data.yaml must declare names")
    normalized: list[tuple[int, str]] = []
    for raw_id, raw_name in value.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int | str):
            raise DatasetPreparationError("source class IDs must be integers")
        try:
            class_id = int(raw_id)
        except ValueError as error:
            raise DatasetPreparationError("source class IDs must be integers") from error
        if str(class_id) != str(raw_id) or not isinstance(raw_name, str):
            raise DatasetPreparationError("source class map is invalid")
        normalized.append((class_id, raw_name))
    return tuple(sorted(normalized))


def _load_and_validate_source_yaml(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DatasetPreparationError("source data.yaml must be a non-symlink regular file")
    try:
        import yaml

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError as error:
        raise DatasetPreparationError(
            "PyYAML is required; run through the pinned uv environment"
        ) from error
    except Exception as error:
        raise DatasetPreparationError(
            "source data.yaml must be readable valid UTF-8 YAML"
        ) from error
    if not isinstance(document, Mapping):
        raise DatasetPreparationError("source data.yaml root must be a mapping")
    if _normalized_class_map(document.get("names")) != SOURCE_CLASS_MAP:
        raise DatasetPreparationError(
            "source class map must exactly match reviewed Construction Site Safety v27 IDs 0-9"
        )
    if document.get("nc") not in (None, len(SOURCE_CLASS_MAP)):
        raise DatasetPreparationError("source data.yaml nc must equal 10 when declared")
    metadata = document.get("roboflow")
    if not isinstance(metadata, Mapping):
        raise DatasetPreparationError("source data.yaml must include Roboflow provenance")
    if metadata.get("version") != SOURCE_VERSION:
        raise DatasetPreparationError("source Roboflow version must equal 27")
    if metadata.get("license") != SOURCE_LICENSE:
        raise DatasetPreparationError("source Roboflow license must equal CC BY 4.0")
    if metadata.get("url") != SOURCE_URL:
        raise DatasetPreparationError(
            "source Roboflow URL must identify Construction Site Safety v27"
        )
    return document


def _relative_key(path: Path, root: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(root).with_suffix("").as_posix())


def _indexed_files(directory: Path, *, images: bool) -> dict[PurePosixPath, Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetPreparationError(f"required directory is missing or unsafe: {directory}")
    indexed: dict[PurePosixPath, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DatasetPreparationError(f"dataset files must not be symlinks: {path}")
        if not path.is_file():
            continue
        if images and path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise DatasetPreparationError(f"unsupported file in images directory: {path}")
        if not images and path.suffix.lower() != ".txt":
            raise DatasetPreparationError(f"unsupported file in labels directory: {path}")
        key = _relative_key(path, directory)
        if key in indexed:
            raise DatasetPreparationError(f"duplicate paired-file stem: {key.as_posix()}")
        indexed[key] = path
    return indexed


def _build_split_plan(root: Path, canonical_name: str, source_name: str) -> SplitPlan:
    split_root = root / source_name
    images = _indexed_files(split_root / "images", images=True)
    labels = _indexed_files(split_root / "labels", images=False)
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels:
        raise DatasetPreparationError(
            f"{source_name} image has no paired label: {missing_labels[0].as_posix()}"
        )
    if missing_images:
        raise DatasetPreparationError(
            f"{source_name} label has no paired image: {missing_images[0].as_posix()}"
        )
    if not images:
        raise DatasetPreparationError(f"{source_name} split must contain at least one image pair")
    pairs = tuple((images[key], labels[key], key) for key in sorted(images))
    return SplitPlan(canonical_name=canonical_name, source_name=source_name, pairs=pairs)


def _split_plans(root: Path) -> tuple[SplitPlan, ...]:
    has_valid = (root / "valid").exists()
    has_val = (root / "val").exists()
    if has_valid == has_val:
        raise DatasetPreparationError(
            "source must contain exactly one validation split: valid or val"
        )
    validation = "valid" if has_valid else "val"
    return (
        _build_split_plan(root, "train", "train"),
        _build_split_plan(root, "val", validation),
        _build_split_plan(root, "test", "test"),
    )


def _parse_and_remap_label(
    path: Path,
) -> tuple[str, Counter[str], Counter[str], int, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DatasetPreparationError(f"label must be readable UTF-8: {path}") from error
    retained: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    output: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise DatasetPreparationError(f"malformed YOLO row at {path}:{line_number}")
        raw_id = fields[0]
        if not _CLASS_ID_PATTERN.fullmatch(raw_id):
            raise DatasetPreparationError(f"invalid class ID at {path}:{line_number}")
        source_id = int(raw_id)
        if source_id >= len(SOURCE_CLASS_MAP):
            raise DatasetPreparationError(f"unknown class ID {source_id} at {path}:{line_number}")
        coordinates: list[float] = []
        for value in fields[1:]:
            try:
                coordinate = float(value)
            except ValueError as error:
                raise DatasetPreparationError(
                    f"non-numeric YOLO coordinate at {path}:{line_number}"
                ) from error
            if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
                raise DatasetPreparationError(
                    f"YOLO coordinates must be finite and normalized at {path}:{line_number}"
                )
            coordinates.append(coordinate)
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise DatasetPreparationError(
                f"YOLO width and height must be positive at {path}:{line_number}"
            )
        source_name = SOURCE_CLASS_MAP[source_id][1]
        canonical_id = SOURCE_TO_CANONICAL.get(source_id)
        if canonical_id is None:
            if source_id not in DROPPED_SOURCE_IDS:
                raise DatasetPreparationError(
                    f"unreviewed class ID {source_id} at {path}:{line_number}"
                )
            dropped[source_name] += 1
            continue
        canonical_name = CANONICAL_CLASS_MAP[canonical_id][1]
        retained[canonical_name] += 1
        output.append(f"{canonical_id} {' '.join(fields[1:])}")
    rendered = "\n".join(output) + ("\n" if output else "")
    return rendered, retained, dropped, sum(retained.values()), sum(dropped.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_facts() -> dict[str, object]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments], check=False, capture_output=True, text=True, timeout=10
        )

    try:
        sha = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commitSha": None, "dirty": None}
    if sha.returncode != 0 or status.returncode != 0:
        return {"available": False, "commitSha": None, "dirty": None}
    return {
        "available": True,
        "commitSha": sha.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def _canonical_data_yaml() -> str:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in CANONICAL_CLASS_MAP)
    return (
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 5\n"
        f"names:\n{names}\n"
    )


def _aggregate_hash(output_files: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(
        list(output_files), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_dataset(input_root: Path, output_root: Path) -> Path:
    """Validate all source bytes and atomically publish one prepared dataset."""

    source_yaml = input_root / "data.yaml"
    _load_and_validate_source_yaml(source_yaml)
    plans = _split_plans(input_root)
    source_paths = [source_yaml]
    for plan in plans:
        for image, label, _key in plan.pairs:
            source_paths.extend((image, label))
    input_hashes = {path: _sha256(path) for path in source_paths}

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    prepared_files: list[PreparedFile] = []
    retained_total: Counter[str] = Counter()
    dropped_total: Counter[str] = Counter()
    split_summaries: dict[str, object] = {}
    try:
        data_yaml = temporary / "data.yaml"
        data_yaml.write_text(_canonical_data_yaml(), encoding="utf-8", newline="\n")
        prepared_files.append(
            PreparedFile(
                input_path="data.yaml",
                output_path="data.yaml",
                input_sha256=input_hashes[source_yaml],
                output_sha256=_sha256(data_yaml),
                input_size_bytes=source_yaml.stat().st_size,
                output_size_bytes=data_yaml.stat().st_size,
            )
        )
        for plan in plans:
            split_retained: Counter[str] = Counter()
            split_dropped: Counter[str] = Counter()
            retained_rows = 0
            dropped_rows = 0
            for source_image, source_label, key in plan.pairs:
                image_rel = (
                    Path(plan.canonical_name)
                    / "images"
                    / Path(f"{key.as_posix()}{source_image.suffix.lower()}")
                )
                label_rel = Path(plan.canonical_name) / "labels" / Path(f"{key.as_posix()}.txt")
                output_image = temporary / image_rel
                output_label = temporary / label_rel
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                rendered, retained, dropped, kept, removed = _parse_and_remap_label(source_label)
                shutil.copyfile(source_image, output_image)
                output_label.write_text(rendered, encoding="utf-8", newline="\n")
                if _sha256(output_image) != input_hashes[source_image]:
                    raise DatasetPreparationError(f"copied image hash mismatch: {source_image}")
                split_retained.update(retained)
                split_dropped.update(dropped)
                retained_rows += kept
                dropped_rows += removed
                prepared_files.extend(
                    (
                        PreparedFile(
                            input_path=source_image.relative_to(input_root).as_posix(),
                            output_path=image_rel.as_posix(),
                            input_sha256=input_hashes[source_image],
                            output_sha256=_sha256(output_image),
                            input_size_bytes=source_image.stat().st_size,
                            output_size_bytes=output_image.stat().st_size,
                        ),
                        PreparedFile(
                            input_path=source_label.relative_to(input_root).as_posix(),
                            output_path=label_rel.as_posix(),
                            input_sha256=input_hashes[source_label],
                            output_sha256=_sha256(output_label),
                            input_size_bytes=source_label.stat().st_size,
                            output_size_bytes=output_label.stat().st_size,
                        ),
                    )
                )
            retained_total.update(split_retained)
            dropped_total.update(split_dropped)
            split_summaries[plan.canonical_name] = {
                "sourceName": plan.source_name,
                "imageCount": len(plan.pairs),
                "labelCount": len(plan.pairs),
                "retainedAnnotationCount": retained_rows,
                "droppedAnnotationCount": dropped_rows,
                "retainedByClass": dict(sorted(split_retained.items())),
                "droppedByClass": dict(sorted(split_dropped.items())),
            }

        for source_path, expected in input_hashes.items():
            if _sha256(source_path) != expected:
                raise DatasetPreparationError(f"source changed during preparation: {source_path}")
        output_files = [
            {
                "path": item.output_path,
                "sha256": item.output_sha256,
                "sizeBytes": item.output_size_bytes,
            }
            for item in sorted(prepared_files, key=lambda item: item.output_path)
        ]
        input_files = [
            {
                "path": path.relative_to(input_root).as_posix(),
                "sha256": digest,
                "sizeBytes": path.stat().st_size,
            }
            for path, digest in sorted(
                input_hashes.items(), key=lambda item: item[0].relative_to(input_root).as_posix()
            )
        ]
        manifest = {
            "schemaVersion": "1.0.0",
            "status": "COMPLETE",
            "source": {
                "url": SOURCE_URL,
                "version": SOURCE_VERSION,
                "license": SOURCE_LICENSE,
                "attribution": SOURCE_ATTRIBUTION,
                "exportFormat": "YOLO",
                "inputRoot": str(input_root),
                "classMap": {str(key): value for key, value in SOURCE_CLASS_MAP},
            },
            "canonical": {
                "outputRoot": str(output_root),
                "classMap": {str(key): value for key, value in CANONICAL_CLASS_MAP},
            },
            "mapping": {
                SOURCE_CLASS_MAP[source_id][1]: CANONICAL_CLASS_MAP[target_id][1]
                for source_id, target_id in sorted(SOURCE_TO_CANONICAL.items())
            },
            "droppedClasses": [
                SOURCE_CLASS_MAP[class_id][1] for class_id in sorted(DROPPED_SOURCE_IDS)
            ],
            "counts": {
                "retainedByClass": dict(sorted(retained_total.items())),
                "droppedByClass": dict(sorted(dropped_total.items())),
            },
            "splits": split_summaries,
            "inputFiles": input_files,
            "outputFiles": output_files,
            "outputAggregate": {
                "algorithm": "sha256-canonical-json-output-files-v1",
                "sha256": _aggregate_hash(output_files),
            },
            "git": _git_facts(),
        }
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root / _MANIFEST_FILENAME


def run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        input_root = _absolute_existing_directory(parsed.input_dir, label="input directory")
        output_root = _validated_output(parsed.output_dir, input_root)
        manifest = prepare_dataset(input_root, output_root)
    except KeyboardInterrupt:
        print("dataset preparation interrupted; no COMPLETE dataset was published", file=sys.stderr)
        return 130
    except (DatasetPreparationError, OSError, RuntimeError) as error:
        print(f"dataset preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"dataset preparation complete: {manifest}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
