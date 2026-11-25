"""Fail-closed experimental annotation scopes; does not authorize model release."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from smartsite_ai.training.dataset_integrity import DatasetIntegrityError

SCOPES_FILENAME = "supervision.scopes.jsonl"
_FIELDS = {"path", "imageSha256", "scope", "positiveOnlyClassId", "sourceId", "labelOrigin"}


def validate_supervision_scopes(root: Path, entries: Sequence[Mapping[str, object]]) -> None:
    """Require one byte-bound scope and corresponding labels for every split image."""
    inventory = {str(entry["path"]): str(entry["sha256"]) for entry in entries}
    if SCOPES_FILENAME not in inventory:
        raise DatasetIntegrityError("experimental dataset requires pinned supervision scopes")
    images = {
        path: digest
        for path, digest in inventory.items()
        if len(PurePosixPath(path).parts) == 3
        and PurePosixPath(path).parts[0] in {"train", "val", "test"}
        and PurePosixPath(path).parts[1] == "images"
    }
    if any("images" in PurePosixPath(path).parts and path not in images for path in inventory):
        raise DatasetIntegrityError("experimental images must use flat split/images paths")
    if not images or not all(any(p.startswith(s + "/") for p in images) for s in ("train", "val")):
        raise DatasetIntegrityError("experimental scopes require train and val images")
    expected_labels = {
        str(PurePosixPath(path).parent.parent / "labels" / (PurePosixPath(path).stem + ".txt"))
        for path in images
    }
    labels = {
        p
        for p in inventory
        if len(PurePosixPath(p).parts) >= 2 and PurePosixPath(p).parts[1] == "labels"
    }
    if labels != expected_labels or len(expected_labels) != len(images):
        raise DatasetIntegrityError("experimental images and labels must correspond one-to-one")
    seen: set[str] = set()
    try:
        with (root / SCOPES_FILENAME).open(encoding="utf-8") as handle:
            for _ in range(len(images) + 1):
                line = handle.readline(16_385)
                if not line:
                    break
                if len(line) > 16_384:
                    raise DatasetIntegrityError("supervision scope row exceeds size limit")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != _FIELDS:
                    raise DatasetIntegrityError("invalid supervision scope fields")
                path = row["path"]
                if not isinstance(path, str) or path not in images or path in seen:
                    raise DatasetIntegrityError("supervision scope image is unknown or repeated")
                if row["imageSha256"] != images[path]:
                    raise DatasetIntegrityError("supervision scope image digest is stale")
                scope, mode = row["scope"], row["positiveOnlyClassId"]
                if (
                    not isinstance(scope, list)
                    or len(scope) != 15
                    or any(type(value) is not bool for value in scope)
                    or not any(scope)
                ):
                    raise DatasetIntegrityError("supervision scope must contain 15 strict booleans")
                if type(mode) is not int or not -1 <= mode < 15:
                    raise DatasetIntegrityError("invalid positive-only class")
                if mode >= 0 and (not scope[mode] or sum(scope) != 1):
                    raise DatasetIntegrityError("positive-only scope must admit exactly its class")
                if any(
                    not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 1_024
                    for key in ("sourceId", "labelOrigin")
                ):
                    raise DatasetIntegrityError("supervision requires source and label provenance")
                label = (
                    PurePosixPath(path).parent.parent
                    / "labels"
                    / (PurePosixPath(path).stem + ".txt")
                )
                _validate_scoped_labels(root / label, scope, mode)
                seen.add(path)
            if seen != set(images) or handle.read(1):
                raise DatasetIntegrityError(
                    "supervision scopes must cover every image exactly once"
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError("supervision scopes must be readable UTF-8 JSONL") from error


def _validate_scoped_labels(path: Path, scope: list[bool], mode: int) -> None:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if not values:
                continue
            try:
                class_id = int(values[0])
                coords = [float(value) for value in values[1:]]
            except (ValueError, IndexError) as error:
                raise DatasetIntegrityError("invalid scoped detection label") from error
            if (
                len(values) != 5
                or str(class_id) != values[0]
                or not 0 <= class_id < 15
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords)
                or coords[2] <= 0
                or coords[3] <= 0
            ):
                raise DatasetIntegrityError("invalid scoped detection label")
            if not scope[class_id]:
                raise DatasetIntegrityError("positive label outside declared source scope")
            count += 1
    if mode >= 0 and count == 0:
        raise DatasetIntegrityError("positive-only image requires an admitted positive label")
