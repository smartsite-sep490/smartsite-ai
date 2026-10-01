"""Deterministic dataset inventories shared by preparation and training."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

PREPARATION_MANIFEST = "preparation.manifest.json"
_SHA256_LENGTH = 64
_CANONICAL_CLASS_MAP = {
    "0": "Person",
    "1": "Hardhat",
    "2": "NO-Hardhat",
    "3": "Safety Vest",
    "4": "NO-Safety Vest",
}


class DatasetIntegrityError(ValueError):
    """Raised when a dataset inventory or its bytes are inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_inventory(files: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(
        list(files), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_link_like(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except AttributeError:
        attributes = 0
    except OSError as error:
        raise DatasetIntegrityError(f"cannot inspect path boundary: {path}") from error
    is_junction = bool(getattr(os.path, "isjunction", lambda _value: False)(path))
    return (
        path.is_symlink()
        or is_junction
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def safe_contained_path(path: Path, root: Path, *, kind: str) -> Path:
    """Reject lexical/resolved escapes and link-like components below root."""

    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise DatasetIntegrityError(f"{kind} escapes dataset root: {path}") from error
    current = root
    for component in relative.parts:
        current /= component
        if is_link_like(current):
            raise DatasetIntegrityError(
                f"{kind} must not use symlink/junction/reparse paths: {path}"
            )
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetIntegrityError(f"{kind} resolves outside dataset root: {path}") from error
    return resolved


def inventory_record(path: Path, root: Path) -> dict[str, object]:
    resolved = safe_contained_path(path, root, kind="dataset file")
    if not resolved.is_file():
        raise DatasetIntegrityError(f"dataset inventory item is not a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "sizeBytes": resolved.stat().st_size,
    }


def walk_safe_files(directory: Path, boundary: Path) -> Iterator[Path]:
    """Walk without following link-like directories, validating before descent."""

    safe_contained_path(directory, boundary, kind="dataset directory")
    for current_text, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        safe_contained_path(current, boundary, kind="dataset directory")
        for name in list(directory_names):
            child = current / name
            if is_link_like(child):
                raise DatasetIntegrityError(
                    f"dataset directory must not be a symlink/junction/reparse point: {child}"
                )
            safe_contained_path(child, boundary, kind="dataset directory")
        for name in sorted(file_names):
            child = current / name
            safe_contained_path(child, boundary, kind="dataset file")
            if not child.is_file():
                raise DatasetIntegrityError(f"dataset item is not a regular file: {child}")
            yield child


def _manifest_entries(payload: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    raw = payload.get("outputFiles")
    if not isinstance(raw, list) or not raw:
        raise DatasetIntegrityError("preparation manifest outputFiles must be non-empty")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "sizeBytes"}:
            raise DatasetIntegrityError("preparation manifest contains an invalid output file")
        relative, digest, size = item.get("path"), item.get("sha256"), item.get("sizeBytes")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise DatasetIntegrityError("preparation manifest contains an invalid output path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in seen:
            raise DatasetIntegrityError(
                "preparation manifest output paths must be unique and local"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DatasetIntegrityError("preparation manifest contains an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DatasetIntegrityError("preparation manifest contains an invalid file size")
        seen.add(relative)
        entries.append({"path": relative, "sha256": digest, "sizeBytes": size})
    return tuple(sorted(entries, key=lambda entry: str(entry["path"])))


def verify_prepared_dataset(data_config: Path) -> str:
    """Verify every prepared byte and return the pinned aggregate SHA-256."""

    root = data_config.parent.resolve(strict=True)
    manifest_path = root / PREPARATION_MANIFEST
    if is_link_like(manifest_path) or not manifest_path.is_file():
        raise DatasetIntegrityError("prepared dataset requires preparation.manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError("preparation manifest must be valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise DatasetIntegrityError("preparation manifest root must be an object")
    if payload.get("schemaVersion") != "1.0.0" or payload.get("status") != "COMPLETE":
        raise DatasetIntegrityError("preparation manifest must be COMPLETE schema 1.0.0")
    canonical = payload.get("canonical")
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("outputRoot") != str(root)
        or canonical.get("classMap") != _CANONICAL_CLASS_MAP
    ):
        raise DatasetIntegrityError("preparation manifest output root does not match dataset")
    entries = _manifest_entries(payload)
    declared = {str(entry["path"]) for entry in entries}
    actual: set[str] = set()
    for path in walk_safe_files(root, root):
        if path.name != PREPARATION_MANIFEST:
            actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        raise DatasetIntegrityError("prepared dataset files differ from preparation manifest")
    verified: list[dict[str, object]] = []
    for entry in entries:
        relative = str(entry["path"])
        path = root.joinpath(*PurePosixPath(relative).parts)
        record = inventory_record(path, root)
        if record != entry:
            raise DatasetIntegrityError(f"prepared dataset file changed: {relative}")
        verified.append(record)
    aggregate = aggregate_inventory(verified)
    output_aggregate = payload.get("outputAggregate")
    if (
        not isinstance(output_aggregate, Mapping)
        or output_aggregate.get("algorithm") != "sha256-canonical-json-output-files-v1"
        or output_aggregate.get("sha256") != aggregate
    ):
        raise DatasetIntegrityError("preparation manifest aggregate does not match dataset")
    return aggregate
