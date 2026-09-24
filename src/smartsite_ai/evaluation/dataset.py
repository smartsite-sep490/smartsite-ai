"""Dataset loading, validation, and integrity checking for PPE evaluation."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from smartsite_ai.evaluation.models import EvaluationFrame, EvaluationManifest


class DatasetValidationError(Exception):
    """Raised when an evaluation dataset manifest or split index fails validation."""


@dataclass(frozen=True)
class LoadedEvaluationDataset:
    """Immutable representation of a validated evaluation dataset loaded into memory."""

    manifest: EvaluationManifest
    splits: dict[str, tuple[EvaluationFrame, ...]]
    dataset_root: Path

    def get_split(self, name: str) -> tuple[EvaluationFrame, ...]:
        if name not in self.splits:
            raise KeyError(
                f"Split '{name}' not found in dataset (available: {tuple(self.splits.keys())})"
            )
        return self.splits[name]

    def all_frames(self) -> tuple[EvaluationFrame, ...]:
        accumulated: list[EvaluationFrame] = []
        for split_name in sorted(self.splits.keys()):
            accumulated.extend(self.splits[split_name])
        return tuple(accumulated)


def compute_dataset_aggregate_sha256(
    splits: dict[str, Sequence[EvaluationFrame]],
) -> str:
    """Compute deterministic aggregate SHA-256 across all splits and frames."""
    hasher = hashlib.sha256()
    for split_name in sorted(splits.keys()):
        hasher.update(split_name.encode("utf-8"))
        for frame in splits[split_name]:
            hasher.update(frame.sha256.lower().encode("utf-8"))
            hasher.update(frame.frame_id.encode("utf-8"))
    return hasher.hexdigest()


def load_evaluation_dataset(
    path: Path | str,
    *,
    verify_checksums: bool = True,
    dataset_root: Path | None = None,
) -> LoadedEvaluationDataset:
    """Load and validate an evaluation dataset manifest and all its referenced split indexes."""
    manifest_path = Path(path).resolve()
    if not manifest_path.exists() or not manifest_path.is_file():
        raise DatasetValidationError(f"Manifest file does not exist: {manifest_path}")

    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = EvaluationManifest.model_validate(manifest_raw)
    except ValidationError as exc:
        raise DatasetValidationError(f"Invalid evaluation manifest schema: {exc}") from exc
    except Exception as exc:
        raise DatasetValidationError(f"Failed to read evaluation manifest: {exc}") from exc

    root = Path(dataset_root).resolve() if dataset_root is not None else manifest_path.parent
    valid_classes = set(manifest.class_map.values())

    parsed_splits: dict[str, tuple[EvaluationFrame, ...]] = {}
    all_seen_frame_ids: set[str] = set()
    all_seen_annotation_ids: set[str] = set()
    split_hashes: dict[str, set[str]] = {}

    for split_name, split_rel_path in sorted(manifest.splits.items()):
        rel_split = Path(split_rel_path)
        if rel_split.is_absolute():
            raise DatasetValidationError(
                f"Split '{split_name}' uses path traversal or absolute path: {split_rel_path}"
            )
        resolved_split = (root / rel_split).resolve()
        if not resolved_split.is_relative_to(root):
            raise DatasetValidationError(
                f"Split '{split_name}' uses path traversal or absolute path: {split_rel_path}"
            )
        if not resolved_split.exists() or not resolved_split.is_file():
            raise DatasetValidationError(f"Split file does not exist: {resolved_split}")

        split_frames: list[EvaluationFrame] = []
        split_hashes[split_name] = set()

        try:
            with resolved_split.open("r", encoding="utf-8") as f:
                for line_idx, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        row_data = json.loads(line)
                        frame = EvaluationFrame.model_validate(row_data)
                    except ValidationError as exc:
                        raise DatasetValidationError(
                            f"Validation error in {resolved_split}:{line_idx}: {exc}"
                        ) from exc
                    except Exception as exc:
                        raise DatasetValidationError(
                            f"Malformed JSON in {resolved_split}:{line_idx}: {exc}"
                        ) from exc

                    # Path safety check for mediaPath
                    media_rel = Path(frame.media_path)
                    if media_rel.is_absolute():
                        raise DatasetValidationError(
                            f"Frame '{frame.frame_id}' in split '{split_name}' mediaPath uses "
                            f"path traversal or absolute path: {frame.media_path}"
                        )
                    resolved_media = (root / media_rel).resolve()
                    if not resolved_media.is_relative_to(root):
                        raise DatasetValidationError(
                            f"Frame '{frame.frame_id}' in split '{split_name}' mediaPath uses "
                            f"path traversal or absolute path: {frame.media_path}"
                        )

                    if verify_checksums:
                        if not resolved_media.exists() or not resolved_media.is_file():
                            raise DatasetValidationError(
                                f"Media file does not exist: {resolved_media} "
                                f"(frame '{frame.frame_id}')"
                            )
                        actual_sha256 = hashlib.sha256(resolved_media.read_bytes()).hexdigest()
                        if actual_sha256 != frame.sha256.lower():
                            raise DatasetValidationError(
                                f"Media file SHA-256 mismatch for frame '{frame.frame_id}': "
                                f"declared {frame.sha256}, actual {actual_sha256}"
                            )

                    # Check duplicate frameId
                    if frame.frame_id in all_seen_frame_ids:
                        raise DatasetValidationError(
                            f"Duplicate frame ID '{frame.frame_id}' found in dataset"
                        )
                    all_seen_frame_ids.add(frame.frame_id)

                    # Validate annotations
                    for ann in frame.annotations:
                        if ann.class_name not in valid_classes:
                            raise DatasetValidationError(
                                f"Unknown class '{ann.class_name}' in frame '{frame.frame_id}' "
                                f"(declared class map: {sorted(valid_classes)})"
                            )
                        if ann.annotation_id in all_seen_annotation_ids:
                            raise DatasetValidationError(
                                f"Duplicate annotation ID '{ann.annotation_id}' found in dataset"
                            )
                        all_seen_annotation_ids.add(ann.annotation_id)

                    split_frames.append(frame)
                    split_hashes[split_name].add(frame.sha256.lower())
        except DatasetValidationError:
            raise
        except Exception as exc:
            raise DatasetValidationError(f"Failed to read split '{split_name}': {exc}") from exc

        parsed_splits[split_name] = tuple(split_frames)

    # Check for exact duplicate hashes crossing splits
    split_names = sorted(split_hashes.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1 :]:
            intersection = split_hashes[s1].intersection(split_hashes[s2])
            if intersection:
                sample_hash = sorted(intersection)[0]
                raise DatasetValidationError(
                    f"Exact duplicate media SHA-256 found crossing splits "
                    f"'{s1}' and '{s2}': {sample_hash}"
                )

    # Check aggregate SHA-256
    if verify_checksums:
        computed_aggregate = compute_dataset_aggregate_sha256(parsed_splits)
        if computed_aggregate != manifest.aggregate_sha256.lower():
            raise DatasetValidationError(
                f"Dataset aggregate SHA-256 mismatch: "
                f"declared {manifest.aggregate_sha256}, computed {computed_aggregate}"
            )

    return LoadedEvaluationDataset(
        manifest=manifest,
        splits=parsed_splits,
        dataset_root=root,
    )
