"""Dataset loading, validation, and integrity checking for PPE evaluation."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import jcs
from pydantic import ValidationError

from smartsite_ai.evaluation.models import EvaluationFrame, EvaluationManifest

MAX_MANIFEST_BYTES: int = 256 * 1024
MAX_JSONL_LINE_BYTES: int = 512 * 1024
MAX_FRAMES_PER_SPLIT: int = 100_000
FILE_HASH_CHUNK_BYTES: int = 1024 * 1024


class DatasetValidationError(Exception):
    """Raised when an evaluation dataset manifest or split index fails validation."""


@dataclass(frozen=True)
class LoadedEvaluationDataset:
    """Immutable representation of a validated evaluation dataset loaded into memory."""

    manifest: EvaluationManifest
    splits: Mapping[str, tuple[EvaluationFrame, ...]]
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


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: '{key}'")
        result[key] = value
    return result


def _resolve_dataset_path(root: Path, raw_path: str, *, label: str) -> Path:
    if "\x00" in raw_path:
        raise DatasetValidationError(f"{label} contains a NUL byte")
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(raw_path)
    if (
        windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or posix_path.is_absolute()
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    ):
        raise DatasetValidationError(f"{label} uses path traversal or absolute path: {raw_path}")
    resolved = (root / Path(raw_path)).resolve()
    if not resolved.is_relative_to(root):
        raise DatasetValidationError(f"{label} uses path traversal or absolute path: {raw_path}")
    return resolved


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(FILE_HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _frame_sort_key(frame: EvaluationFrame) -> tuple[str, int, int, str]:
    """Order each media source deterministically, preserving video timeline order."""
    is_video = int(frame.frame_index is not None)
    frame_index = frame.frame_index if frame.frame_index is not None else -1
    return (frame.media_path, is_video, frame_index, frame.frame_id)


def compute_dataset_aggregate_sha256(
    splits: Mapping[str, Sequence[EvaluationFrame]],
) -> str:
    """Compute deterministic aggregate SHA-256 over canonicalized dataset contents."""
    canonical_representation: dict[str, list[dict[str, Any]]] = {}
    for split_name in sorted(splits.keys()):
        sorted_frames = sorted(splits[split_name], key=_frame_sort_key)
        frames_list = []
        for frame in sorted_frames:
            sorted_anns = sorted(frame.annotations, key=lambda a: a.annotation_id)
            frame_dict = {
                "frameId": frame.frame_id,
                "mediaPath": frame.media_path,
                "sha256": frame.sha256.lower(),
                "width": frame.width,
                "height": frame.height,
                "frameIndex": frame.frame_index,
                "videoTimeSeconds": frame.video_time_seconds,
                "annotations": [
                    {
                        "annotationId": a.annotation_id,
                        "className": a.class_name,
                        "boundingBox": {
                            "x1": a.bounding_box.x1,
                            "y1": a.bounding_box.y1,
                            "x2": a.bounding_box.x2,
                            "y2": a.bounding_box.y2,
                        },
                        "personInstanceId": a.person_instance_id,
                        "relatedPersonAnnotationId": a.related_person_annotation_id,
                        "visibility": a.visibility,
                        "observablePpeItems": a.observable_ppe_items,
                    }
                    for a in sorted_anns
                ],
            }
            frames_list.append(frame_dict)
        canonical_representation[split_name] = frames_list

    canonical_bytes = jcs.canonicalize(canonical_representation)
    if canonical_bytes is None:
        raise DatasetValidationError("Failed to canonicalize dataset content as RFC 8785 JSON")
    return hashlib.sha256(canonical_bytes).hexdigest()


def load_evaluation_dataset(
    path: Path | str,
) -> LoadedEvaluationDataset:
    """Load and validate an evaluation dataset manifest and all its referenced split indexes."""
    manifest_path = Path(path).resolve()
    if not manifest_path.exists() or not manifest_path.is_file():
        raise DatasetValidationError(f"Manifest file does not exist: {manifest_path}")

    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise DatasetValidationError(
            f"Manifest file exceeds maximum allowed size ({MAX_MANIFEST_BYTES} bytes)"
        )

    try:
        manifest_raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
        manifest = EvaluationManifest.model_validate(manifest_raw)
    except ValidationError as exc:
        raise DatasetValidationError(f"Invalid evaluation manifest schema: {exc}") from exc
    except Exception as exc:
        raise DatasetValidationError(f"Failed to read evaluation manifest: {exc}") from exc

    root = manifest_path.parent.resolve()
    valid_classes = set(manifest.class_map.values())

    parsed_splits: dict[str, tuple[EvaluationFrame, ...]] = {}
    all_seen_frame_ids: set[str] = set()
    all_seen_annotation_ids: set[str] = set()
    split_hashes: dict[str, set[str]] = {}
    media_checksum_cache: dict[Path, str] = {}
    seen_video_frames: set[tuple[Path, int]] = set()
    video_positions: dict[Path, list[tuple[int, float, str]]] = {}

    for split_name, split_rel_path in sorted(manifest.splits.items()):
        resolved_split = _resolve_dataset_path(
            root,
            split_rel_path,
            label=f"Split '{split_name}'",
        )
        if not resolved_split.exists() or not resolved_split.is_file():
            raise DatasetValidationError(f"Split file does not exist: {resolved_split}")

        split_frames: list[EvaluationFrame] = []
        split_hashes[split_name] = set()

        try:
            with resolved_split.open("rb") as f:
                line_idx = 0
                while True:
                    raw_bytes = f.readline(MAX_JSONL_LINE_BYTES + 1)
                    if not raw_bytes:
                        break
                    line_idx += 1
                    if len(raw_bytes) > MAX_JSONL_LINE_BYTES:
                        raise DatasetValidationError(
                            f"Line {line_idx} in {resolved_split} exceeds maximum allowed size "
                            f"({MAX_JSONL_LINE_BYTES} bytes)"
                        )
                    line_str = raw_bytes.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        row_data = json.loads(
                            line_str,
                            object_pairs_hook=_reject_duplicate_object_keys,
                        )
                        frame = EvaluationFrame.model_validate(row_data)
                    except ValidationError as exc:
                        raise DatasetValidationError(
                            f"Validation error in {resolved_split}:{line_idx}: {exc}"
                        ) from exc
                    except Exception as exc:
                        raise DatasetValidationError(
                            f"Malformed JSON in {resolved_split}:{line_idx}: {exc}"
                        ) from exc

                    if len(split_frames) >= MAX_FRAMES_PER_SPLIT:
                        raise DatasetValidationError(
                            f"Split '{split_name}' exceeds maximum frame count "
                            f"({MAX_FRAMES_PER_SPLIT})"
                        )

                    # Path safety check for mediaPath
                    resolved_media = _resolve_dataset_path(
                        root,
                        frame.media_path,
                        label=f"Frame '{frame.frame_id}' in split '{split_name}' mediaPath",
                    )

                    # Always check media file exists and is a file
                    if not resolved_media.exists() or not resolved_media.is_file():
                        raise DatasetValidationError(
                            f"Media file does not exist: {resolved_media} "
                            f"(frame '{frame.frame_id}')"
                        )

                    actual_sha256 = media_checksum_cache.get(resolved_media)
                    if actual_sha256 is None:
                        actual_sha256 = _sha256_file(resolved_media)
                        media_checksum_cache[resolved_media] = actual_sha256
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

                    # Validate annotations and index them for cross-referencing
                    frame_ann_by_id = {}
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
                        frame_ann_by_id[ann.annotation_id] = ann

                        if frame.frame_index is not None:
                            if ann.person_instance_id is None:
                                raise DatasetValidationError(
                                    f"Video frame '{frame.frame_id}' annotation "
                                    f"'{ann.annotation_id}' must declare personInstanceId"
                                )
                            if ann.class_name == "Person":
                                if ann.observable_ppe_items is None:
                                    raise DatasetValidationError(
                                        f"Video frame '{frame.frame_id}' Person annotation "
                                        f"'{ann.annotation_id}' must declare observablePpeItems"
                                    )
                            elif ann.related_person_annotation_id is None:
                                raise DatasetValidationError(
                                    f"Video frame '{frame.frame_id}' PPE annotation "
                                    f"'{ann.annotation_id}' must declare relatedPersonAnnotationId"
                                )

                    # Validate relatedPersonAnnotationId within this frame
                    for ann in frame.annotations:
                        if ann.related_person_annotation_id is not None:
                            if ann.related_person_annotation_id == ann.annotation_id:
                                raise DatasetValidationError(
                                    f"Frame '{frame.frame_id}' annotation '{ann.annotation_id}' "
                                    f"cannot reference itself as relatedPersonAnnotationId"
                                )
                            target = frame_ann_by_id.get(ann.related_person_annotation_id)
                            if target is None:
                                raise DatasetValidationError(
                                    f"Frame '{frame.frame_id}' annotation '{ann.annotation_id}' "
                                    f"references non-existent relatedPersonAnnotationId "
                                    f"'{ann.related_person_annotation_id}'"
                                )
                            if target.class_name != "Person":
                                raise DatasetValidationError(
                                    f"Frame '{frame.frame_id}' annotation '{ann.annotation_id}' "
                                    f"references non-Person annotation "
                                    f"'{ann.related_person_annotation_id}' "
                                    f"(class: '{target.class_name}')"
                                )
                            if (
                                ann.person_instance_id is not None
                                and target.person_instance_id is not None
                                and ann.person_instance_id != target.person_instance_id
                            ):
                                raise DatasetValidationError(
                                    f"Frame '{frame.frame_id}' annotation '{ann.annotation_id}' "
                                    f"personInstanceId ({ann.person_instance_id}) does not match "
                                    "related person's personInstanceId "
                                    f"({target.person_instance_id})"
                                )

                    # Sort annotations stably by annotation_id
                    sorted_frame_anns = tuple(
                        sorted(frame.annotations, key=lambda a: a.annotation_id)
                    )
                    sorted_frame = frame.model_copy(update={"annotations": sorted_frame_anns})

                    if frame.frame_index is not None and frame.video_time_seconds is not None:
                        video_identity = (resolved_media, frame.frame_index)
                        if video_identity in seen_video_frames:
                            raise DatasetValidationError(
                                f"Duplicate video frame identity for '{frame.media_path}' "
                                f"at frameIndex {frame.frame_index}"
                            )
                        seen_video_frames.add(video_identity)
                        video_positions.setdefault(resolved_media, []).append(
                            (frame.frame_index, frame.video_time_seconds, frame.frame_id)
                        )

                    split_frames.append(sorted_frame)
                    split_hashes[split_name].add(frame.sha256.lower())
        except DatasetValidationError:
            raise
        except Exception as exc:
            raise DatasetValidationError(f"Failed to read split '{split_name}': {exc}") from exc

        parsed_splits[split_name] = tuple(sorted(split_frames, key=_frame_sort_key))

    for media_path, positions in video_positions.items():
        ordered = sorted(positions, key=lambda item: item[0])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[1] <= previous[1]:
                raise DatasetValidationError(
                    f"Video presentation timestamps for '{media_path}' must increase "
                    f"with frameIndex: frame '{previous[2]}' has {previous[1]}, "
                    f"frame '{current[2]}' has {current[1]}"
                )

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

    computed_aggregate = compute_dataset_aggregate_sha256(parsed_splits)
    if computed_aggregate != manifest.aggregate_sha256.lower():
        raise DatasetValidationError(
            f"Dataset aggregate SHA-256 mismatch: "
            f"declared {manifest.aggregate_sha256}, computed {computed_aggregate}"
        )

    return LoadedEvaluationDataset(
        manifest=manifest,
        splits=MappingProxyType(parsed_splits),
        dataset_root=root,
    )
