"""Build an immutable local temporal-video corpus for MF05 event evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from smartsite_ai.evaluation.dataset import (
    MAX_JSONL_LINE_BYTES,
    DatasetValidationError,
    compute_dataset_aggregate_sha256,
    load_evaluation_dataset,
)
from smartsite_ai.evaluation.execution import (
    MAX_EPISODE_LINE_BYTES,
    load_ground_truth_episodes,
)
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    EvaluationFrame,
    EvaluationManifest,
    GroundTruthPpeEpisode,
)
from smartsite_ai.evaluation.video_corpus import (
    OpenCvVideoProbe,
    ReviewedCorpusSource,
    ReviewedFrame,
    VideoFacts,
    VideoProbe,
    evaluated_camera_seconds,
    source_episode_key,
)
from smartsite_ai.training.dataset_integrity import is_link_like, sha256_file

MAX_SOURCE_MANIFEST_BYTES = 256 * 1024
MAX_REVIEWED_FRAMES = 100_000
MAX_VIDEO_FILE_BYTES = 50 * 1024 * 1024 * 1024
RATE_GATE_SECONDS = 1800.0
_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
_CLASS_MAP = {str(index): name for index, name in enumerate(CANONICAL_PPE_CLASSES)}
_PPE_CLASS = {
    "Hardhat": ("HARD_HAT", "PRESENT"),
    "NO-Hardhat": ("HARD_HAT", "MISSING"),
    "Safety Vest": ("SAFETY_VEST", "PRESENT"),
    "NO-Safety Vest": ("SAFETY_VEST", "MISSING"),
}


class VideoCorpusBuildError(RuntimeError):
    """A safe, operator-actionable corpus build failure."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reviewed local video corpus for YOLO11s PPE event evaluation."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _absolute_regular_file(path: Path, *, label: str, suffixes: frozenset[str]) -> Path:
    if not path.is_absolute():
        raise VideoCorpusBuildError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise VideoCorpusBuildError(f"{label} must exist") from error
    if is_link_like(path) or not resolved.is_file() or resolved.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise VideoCorpusBuildError(f"{label} must be a non-link regular file ({expected})")
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
        raise VideoCorpusBuildError("in-repository output directory must be ignored by Git")


def _new_output_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise VideoCorpusBuildError("output directory must be an absolute path")
    if path.exists() or path.is_symlink():
        raise VideoCorpusBuildError("output directory must not already exist")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise VideoCorpusBuildError("output parent must exist") from error
    if is_link_like(parent) or not parent.is_dir():
        raise VideoCorpusBuildError("output parent must be a non-link directory")
    output = parent / path.name
    _require_ignored_repository_path(output)
    return output


def _load_source_manifest(path: Path) -> tuple[Path, str, ReviewedCorpusSource]:
    path = _absolute_regular_file(path, label="source manifest", suffixes=frozenset({".json"}))
    if path.stat().st_size > MAX_SOURCE_MANIFEST_BYTES:
        raise VideoCorpusBuildError("source manifest exceeds size limit")
    initial_hash = sha256_file(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        source = ReviewedCorpusSource.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise VideoCorpusBuildError(f"invalid reviewed source manifest: {error}") from error
    if sha256_file(path) != initial_hash:
        raise VideoCorpusBuildError("source manifest changed while it was read")
    return path, initial_hash, source


def _load_reviewed_frames(
    path: Path, *, clip_id: str, max_frame_gap_seconds: float
) -> tuple[Path, str, tuple[ReviewedFrame, ...]]:
    path = _absolute_regular_file(
        path, label=f"framesPath for {clip_id}", suffixes=frozenset({".jsonl"})
    )
    initial_hash = sha256_file(path)
    frames: list[ReviewedFrame] = []
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if len(raw_line) > MAX_JSONL_LINE_BYTES:
                    raise VideoCorpusBuildError(
                        f"reviewed frame line {line_number} for {clip_id} exceeds size limit"
                    )
                if not raw_line.strip():
                    raise VideoCorpusBuildError(
                        f"reviewed frame line {line_number} for {clip_id} must not be blank"
                    )
                try:
                    payload = json.loads(
                        raw_line.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
                    )
                    frame = ReviewedFrame.model_validate(payload)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise VideoCorpusBuildError(
                        f"invalid reviewed frame {line_number} for {clip_id}: {error}"
                    ) from error
                if frame.clip_id != clip_id:
                    raise VideoCorpusBuildError(
                        f"reviewed frame {line_number} declares unexpected clipId {frame.clip_id}"
                    )
                frames.append(frame)
                if len(frames) > MAX_REVIEWED_FRAMES:
                    raise VideoCorpusBuildError("reviewed frame count exceeds limit")
    except OSError as error:
        raise VideoCorpusBuildError(f"could not read framesPath for {clip_id}") from error
    if not frames:
        raise VideoCorpusBuildError(f"framesPath for {clip_id} must contain at least one frame")
    indexes = [frame.frame_index for frame in frames]
    timestamps = [frame.video_time_seconds for frame in frames]
    if any(current <= previous for previous, current in zip(indexes, indexes[1:], strict=False)):
        raise VideoCorpusBuildError(f"frameIndex values for {clip_id} must strictly increase")
    if any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise VideoCorpusBuildError(f"videoTimeSeconds values for {clip_id} must strictly increase")
    if any(
        current - previous > max_frame_gap_seconds + 1e-9
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise VideoCorpusBuildError(f"reviewed frame gap for {clip_id} exceeds maxFrameGapSeconds")
    if sha256_file(path) != initial_hash:
        raise VideoCorpusBuildError(f"framesPath for {clip_id} changed while it was read")
    return path, initial_hash, tuple(frames)


def _validate_frame_against_video(
    frame: ReviewedFrame, facts: VideoFacts, *, decoded_time_seconds: float
) -> None:
    if frame.width != facts.width or frame.height != facts.height:
        raise VideoCorpusBuildError(
            f"reviewed dimensions for {frame.clip_id} frame {frame.frame_index} do not match video"
        )
    if frame.frame_index >= facts.frame_count:
        raise VideoCorpusBuildError(
            f"frameIndex {frame.frame_index} for {frame.clip_id} is outside the video"
        )
    frame_duration = facts.duration_seconds / facts.frame_count
    if frame.video_time_seconds > facts.duration_seconds + frame_duration:
        raise VideoCorpusBuildError(
            f"videoTimeSeconds for {frame.clip_id} frame {frame.frame_index} is outside the video"
        )
    tolerance = max(0.05, frame_duration * 1.5)
    if abs(frame.video_time_seconds - decoded_time_seconds) > tolerance:
        raise VideoCorpusBuildError(
            f"videoTimeSeconds for {frame.clip_id} frame {frame.frame_index} "
            "does not identify the decoded video frame"
        )


def _validate_frame_semantics(frame: ReviewedFrame) -> None:
    annotation_by_id = {item.annotation_id: item for item in frame.annotations}
    person_ids = [
        item.person_instance_id
        for item in frame.annotations
        if item.class_name == "Person" and item.person_instance_id is not None
    ]
    if len(person_ids) != len(set(person_ids)):
        raise VideoCorpusBuildError(
            f"{frame.clip_id} frame {frame.frame_index} repeats a Person personInstanceId"
        )
    states: dict[tuple[int | str, str], set[str]] = defaultdict(set)
    for annotation in frame.annotations:
        if annotation.class_name not in _PPE_CLASS:
            continue
        if annotation.related_person_annotation_id is None:
            continue
        person = annotation_by_id.get(annotation.related_person_annotation_id)
        if person is None or person.person_instance_id is None:
            continue
        ppe_item, state = _PPE_CLASS[annotation.class_name]
        if person.observable_ppe_items is not None and ppe_item not in person.observable_ppe_items:
            raise VideoCorpusBuildError(
                f"{frame.clip_id} frame {frame.frame_index} labels {ppe_item} for a person "
                "whose observablePpeItems omits it"
            )
        states[(person.person_instance_id, ppe_item)].add(state)
    conflicts = [key for key, values in states.items() if values == {"PRESENT", "MISSING"}]
    if conflicts:
        person, item = sorted(conflicts, key=lambda value: (str(value[0]), value[1]))[0]
        raise VideoCorpusBuildError(
            f"{frame.clip_id} frame {frame.frame_index} has conflicting {item} labels "
            f"for person {person}"
        )


def _output_frame(frame: ReviewedFrame, *, media_path: str, media_sha256: str) -> EvaluationFrame:
    frame_id = str(
        uuid5(NAMESPACE_URL, f"smartsite-event-evaluation:{frame.clip_id}:{frame.frame_index}")
    )
    try:
        return EvaluationFrame(
            frameId=frame_id,
            mediaPath=media_path,
            sha256=media_sha256,
            width=frame.width,
            height=frame.height,
            frameIndex=frame.frame_index,
            videoTimeSeconds=frame.video_time_seconds,
            annotations=frame.annotations,
        )
    except ValidationError as error:
        raise VideoCorpusBuildError(
            f"invalid reviewed frame {frame.clip_id}:{frame.frame_index}: {error}"
        ) from error


def _negative_evidence(
    frames_by_clip: Mapping[str, Sequence[ReviewedFrame]],
) -> tuple[tuple[str, int | str, str, float], ...]:
    evidence: list[tuple[str, int | str, str, float]] = []
    for clip_id, frames in frames_by_clip.items():
        for frame in frames:
            annotations = {item.annotation_id: item for item in frame.annotations}
            for annotation in frame.annotations:
                mapping = _PPE_CLASS.get(annotation.class_name)
                if mapping is None or mapping[1] != "MISSING":
                    continue
                person = annotations.get(annotation.related_person_annotation_id or "")
                if person is not None and person.person_instance_id is not None:
                    evidence.append(
                        (
                            clip_id,
                            person.person_instance_id,
                            mapping[0],
                            frame.video_time_seconds,
                        )
                    )
    return tuple(evidence)


def _positive_evidence(
    frames_by_clip: Mapping[str, Sequence[ReviewedFrame]],
) -> tuple[tuple[str, int | str, str, float], ...]:
    evidence: list[tuple[str, int | str, str, float]] = []
    for clip_id, frames in frames_by_clip.items():
        for frame in frames:
            annotations = {item.annotation_id: item for item in frame.annotations}
            for annotation in frame.annotations:
                mapping = _PPE_CLASS.get(annotation.class_name)
                if mapping is None or mapping[1] != "PRESENT":
                    continue
                person = annotations.get(annotation.related_person_annotation_id or "")
                if person is not None and person.person_instance_id is not None:
                    evidence.append(
                        (
                            clip_id,
                            person.person_instance_id,
                            mapping[0],
                            frame.video_time_seconds,
                        )
                    )
    return tuple(evidence)


def _validate_and_rewrite_episodes(
    episodes: Sequence[GroundTruthPpeEpisode],
    *,
    frames_by_clip: Mapping[str, Sequence[ReviewedFrame]],
    video_facts: Mapping[str, VideoFacts],
    media_paths: Mapping[str, str],
) -> tuple[GroundTruthPpeEpisode, ...]:
    people_by_clip = {
        clip_id: {
            annotation.person_instance_id
            for frame in frames
            for annotation in frame.annotations
            if annotation.class_name == "Person" and annotation.person_instance_id is not None
        }
        for clip_id, frames in frames_by_clip.items()
    }
    grouped: dict[tuple[str, int | str, str], list[GroundTruthPpeEpisode]] = defaultdict(list)
    for episode in episodes:
        if episode.clip_id not in frames_by_clip:
            raise VideoCorpusBuildError(f"episode references unknown clipId {episode.clip_id}")
        if episode.person_instance_id not in people_by_clip[episode.clip_id]:
            raise VideoCorpusBuildError(
                f"episode references person absent from reviewed clip {episode.clip_id}"
            )
        times = [frame.video_time_seconds for frame in frames_by_clip[episode.clip_id]]
        if episode.start_time_seconds < min(times) or episode.end_time_seconds > max(times):
            raise VideoCorpusBuildError(
                f"episode for {episode.clip_id} is outside the reviewed frame interval"
            )
        if episode.end_time_seconds > video_facts[episode.clip_id].duration_seconds:
            raise VideoCorpusBuildError(f"episode for {episode.clip_id} is outside the video")
        grouped[source_episode_key(episode)].append(episode)

    for key, series in grouped.items():
        ordered = sorted(series, key=lambda item: (item.start_time_seconds, item.end_time_seconds))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.start_time_seconds <= previous.end_time_seconds:
                raise VideoCorpusBuildError(
                    f"episodes overlap for clip/person/PPE series {key[0]}/{key[1]}/{key[2]}"
                )

    evidence = _negative_evidence(frames_by_clip)
    positive_evidence = _positive_evidence(frames_by_clip)
    for episode in episodes:
        matches = [
            point
            for point in evidence
            if point[:3] == source_episode_key(episode)
            and episode.start_time_seconds <= point[3] <= episode.end_time_seconds
        ]
        if not matches:
            raise VideoCorpusBuildError(
                f"episode for {episode.clip_id}/{episode.person_instance_id}/{episode.ppe_item} "
                "has no reviewed negative-PPE frame evidence"
            )
        if any(
            point[:3] == source_episode_key(episode)
            and episode.start_time_seconds <= point[3] <= episode.end_time_seconds
            for point in positive_evidence
        ):
            raise VideoCorpusBuildError(
                f"episode for {episode.clip_id}/{episode.person_instance_id}/{episode.ppe_item} "
                "contains reviewed positive-PPE evidence"
            )
    for point in evidence:
        covering = [
            episode
            for episode in episodes
            if source_episode_key(episode) == point[:3]
            and episode.start_time_seconds <= point[3] <= episode.end_time_seconds
        ]
        if len(covering) != 1:
            raise VideoCorpusBuildError(
                f"negative-PPE evidence at {point[0]}:{point[3]} must belong to exactly one episode"
            )

    return tuple(
        episode.model_copy(update={"clip_id": media_paths[episode.clip_id]})
        for episode in sorted(
            episodes,
            key=lambda item: (
                item.clip_id,
                str(item.person_instance_id),
                item.ppe_item,
                item.start_time_seconds,
                item.end_time_seconds,
            ),
        )
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_frames(path: Path, frames: Sequence[EvaluationFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(frame.model_dump_json(by_alias=True, exclude_none=True) + "\n")


def _write_episodes(path: Path, episodes: Sequence[GroundTruthPpeEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            row = episode.model_dump_json(by_alias=True, exclude_none=True)
            if len(row.encode("utf-8")) > MAX_EPISODE_LINE_BYTES:
                raise VideoCorpusBuildError("generated episode row exceeds size limit")
            handle.write(row + "\n")


def build_video_evaluation_corpus(
    source_manifest: Path,
    output_dir: Path,
    *,
    video_probe: VideoProbe | None = None,
) -> Path:
    """Validate reviewed labels and atomically publish official-gate inputs."""

    source_path, source_hash, source = _load_source_manifest(source_manifest)
    output = _new_output_directory(output_dir)
    episode_source_path = _absolute_regular_file(
        source.episodes_path,
        label="episodesPath",
        suffixes=frozenset({".jsonl"}),
    )
    episodes_hash = sha256_file(episode_source_path)
    episodes = load_ground_truth_episodes(episode_source_path)
    if sha256_file(episode_source_path) != episodes_hash:
        raise VideoCorpusBuildError("episodesPath changed while it was read")
    probe = video_probe or OpenCvVideoProbe()

    inputs: list[tuple[Any, Path, Path, str, tuple[ReviewedFrame, ...], VideoFacts]] = []
    frames_by_clip: dict[str, tuple[ReviewedFrame, ...]] = {}
    facts_by_clip: dict[str, VideoFacts] = {}
    media_paths: dict[str, str] = {}
    clip_by_source_path: dict[Path, str] = {}
    clip_by_media_sha256: dict[str, str] = {}
    total_reviewed_frames = 0
    for clip in source.clips:
        media = _absolute_regular_file(
            clip.media_path, label=f"mediaPath for {clip.clip_id}", suffixes=_VIDEO_SUFFIXES
        )
        duplicate_path_clip = clip_by_source_path.get(media)
        if duplicate_path_clip is not None:
            raise VideoCorpusBuildError(
                f"clips {duplicate_path_clip} and {clip.clip_id} resolve to the same mediaPath"
            )
        if media.stat().st_size > MAX_VIDEO_FILE_BYTES:
            raise VideoCorpusBuildError(f"reviewed video {clip.clip_id} exceeds size limit")
        media_sha256 = sha256_file(media)
        if media_sha256 != clip.sha256:
            raise VideoCorpusBuildError(f"media SHA-256 mismatch for {clip.clip_id}")
        duplicate_content_clip = clip_by_media_sha256.get(media_sha256)
        if duplicate_content_clip is not None:
            raise VideoCorpusBuildError(
                f"clips {duplicate_content_clip} and {clip.clip_id} contain identical media bytes"
            )
        clip_by_source_path[media] = clip.clip_id
        clip_by_media_sha256[media_sha256] = clip.clip_id
        frames_path, frames_hash, frames = _load_reviewed_frames(
            clip.frames_path,
            clip_id=clip.clip_id,
            max_frame_gap_seconds=source.max_frame_gap_seconds,
        )
        total_reviewed_frames += len(frames)
        if total_reviewed_frames > MAX_REVIEWED_FRAMES:
            raise VideoCorpusBuildError("total reviewed frame count exceeds limit")
        facts = probe.probe(media)
        decoded_times = probe.frame_times(media, [frame.frame_index for frame in frames])
        if set(decoded_times) != {frame.frame_index for frame in frames}:
            raise VideoCorpusBuildError(f"video probe omitted reviewed frames for {clip.clip_id}")
        for frame in frames:
            _validate_frame_against_video(
                frame,
                facts,
                decoded_time_seconds=decoded_times[frame.frame_index],
            )
            _validate_frame_semantics(frame)
        media_path = f"media/test/{clip.clip_id}{media.suffix.lower()}"
        inputs.append((clip, media, frames_path, frames_hash, frames, facts))
        frames_by_clip[clip.clip_id] = frames
        facts_by_clip[clip.clip_id] = facts
        media_paths[clip.clip_id] = media_path

    rewritten_episodes = _validate_and_rewrite_episodes(
        episodes,
        frames_by_clip=frames_by_clip,
        video_facts=facts_by_clip,
        media_paths=media_paths,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        output_frames: list[EvaluationFrame] = []
        clip_records: list[dict[str, object]] = []
        for clip, media, frames_path, frames_hash, frames, facts in inputs:
            relative_media = media_paths[clip.clip_id]
            destination = temporary / relative_media
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(media, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != clip.sha256 or sha256_file(media) != clip.sha256:
                raise VideoCorpusBuildError(f"media changed while copying {clip.clip_id}")
            if sha256_file(frames_path) != frames_hash:
                raise VideoCorpusBuildError(f"framesPath changed while copying {clip.clip_id}")
            output_frames.extend(
                _output_frame(frame, media_path=relative_media, media_sha256=copied_hash)
                for frame in frames
            )
            clip_records.append(
                {
                    "clipId": clip.clip_id,
                    "sourceFileName": media.name,
                    "mediaPath": relative_media,
                    "mediaSha256": copied_hash,
                    "framesSha256": frames_hash,
                    "labelledFrameCount": len(frames),
                    "video": {
                        "width": facts.width,
                        "height": facts.height,
                        "frameCount": facts.frame_count,
                        "durationSeconds": facts.duration_seconds,
                    },
                }
            )

        split_frames = {
            "train": (),
            "validation": (),
            "test": tuple(output_frames),
        }
        aggregate = compute_dataset_aggregate_sha256(split_frames)
        manifest = EvaluationManifest(
            schemaVersion="1.0.0",
            datasetId=source.dataset_id,
            datasetVersion=source.dataset_version,
            sourceUrl=source.source_url,
            license=source.license,
            aggregateSha256=aggregate,
            classMap=_CLASS_MAP,
            splits={
                "train": "indexes/train.jsonl",
                "validation": "indexes/validation.jsonl",
                "test": "indexes/test.jsonl",
            },
        )
        for split, frames in split_frames.items():
            _write_frames(temporary / "indexes" / f"{split}.jsonl", frames)
        episodes_path = temporary / "indexes" / "test-episodes.jsonl"
        _write_episodes(episodes_path, rewritten_episodes)
        manifest_path = temporary / "evaluation.manifest.json"
        _write_json(manifest_path, manifest.model_dump(mode="json", by_alias=True))

        duration = evaluated_camera_seconds(frames_by_clip)
        corpus_manifest_path = temporary / "corpus.manifest.json"
        _write_json(
            corpus_manifest_path,
            {
                "schemaVersion": "1.0.0",
                "status": "COMPLETE",
                "review": {
                    "reviewedBy": source.reviewed_by,
                    "reviewedAtUtc": source.reviewed_at_utc.isoformat(),
                },
                "source": {
                    "manifestFileName": source_path.name,
                    "manifestSha256": source_hash,
                    "episodesFileName": episode_source_path.name,
                    "episodesSha256": episodes_hash,
                },
                "clips": clip_records,
                "evaluation": {
                    "manifest": "evaluation.manifest.json",
                    "aggregateSha256": aggregate,
                    "episodesIndex": "indexes/test-episodes.jsonl",
                    "episodeCount": len(rewritten_episodes),
                    "labelledFrameCount": len(output_frames),
                    "evaluatedCameraSeconds": duration,
                    "rateGateEligible": duration >= RATE_GATE_SECONDS,
                    "rateGateIneligibilityReason": (
                        None
                        if duration >= RATE_GATE_SECONDS
                        else "requires at least 1800 evaluated camera-seconds"
                    ),
                },
            },
        )

        loaded = load_evaluation_dataset(manifest_path)
        loaded_episodes = load_ground_truth_episodes(episodes_path)
        if sha256_file(source_path) != source_hash:
            raise VideoCorpusBuildError("source manifest changed during conversion")
        if sha256_file(episode_source_path) != episodes_hash:
            raise VideoCorpusBuildError("episodesPath changed during conversion")
        if loaded.manifest.aggregate_sha256 != aggregate or len(loaded_episodes) != len(
            rewritten_episodes
        ):
            raise VideoCorpusBuildError("published evaluation corpus is inconsistent")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "corpus.manifest.json"


def run(
    argv: Sequence[str] | None = None,
    *,
    video_probe: VideoProbe | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        manifest = build_video_evaluation_corpus(
            parsed.source_manifest,
            parsed.output_dir,
            video_probe=video_probe,
        )
    except KeyboardInterrupt:
        print("video evaluation corpus build interrupted; no output was published", file=sys.stderr)
        return 130
    except (
        DatasetValidationError,
        VideoCorpusBuildError,
        ValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"video evaluation corpus build failed: {error}", file=sys.stderr)
        return 1
    print(f"video evaluation corpus complete: {manifest}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()


__all__ = [
    "VideoCorpusBuildError",
    "build_video_evaluation_corpus",
    "main",
    "run",
]
