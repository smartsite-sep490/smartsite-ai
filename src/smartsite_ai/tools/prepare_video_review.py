"""Build a local-only DRAFT review package from real PPE video frames."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.evaluation.review_package import (
    DraftFrameProposal,
    DraftReviewPackageManifest,
    ProposalDetection,
    ProposalPerson,
    ProposalPpeAssociation,
)
from smartsite_ai.evaluation.runtime_adapters import ultralytics_runtime_sandbox
from smartsite_ai.inference.loading import RunnerFactory, load_yolo11_detector
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.training.dataset_integrity import is_link_like

_VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
_MAX_VIDEOS = 32
_MAX_VIDEO_BYTES = 64 * 1024 * 1024 * 1024
_MAX_SAMPLES = 20_000
_MIN_CADENCE_SECONDS = 0.1
_MAX_CADENCE_SECONDS = 1.0
_REGION_ID = "00000000-0000-4000-8000-000000000001"


class ReviewPackageError(RuntimeError):
    """The requested review package is unsafe, invalid, or incomplete."""


@dataclass(frozen=True, slots=True)
class VideoFacts:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True, slots=True)
class VideoSample:
    frame_index: int
    video_time_seconds: float
    payload: bytes
    drawable: object


class ReviewVideo(Protocol):
    @property
    def facts(self) -> VideoFacts: ...

    def samples(self, cadence_seconds: float) -> Iterator[VideoSample]: ...

    def write_jpeg(self, path: Path, drawable: object) -> None: ...

    def write_overlay(
        self, path: Path, drawable: object, proposal: DraftFrameProposal
    ) -> object: ...

    def close(self) -> None: ...


class OpenCvReviewVideo:
    """Lazy OpenCV reader/writer for deterministic sampled review evidence."""

    def __init__(self, path: Path) -> None:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - selected runtime extra
            raise ReviewPackageError("OpenCV runtime is unavailable") from error
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ReviewPackageError(f"OpenCV could not open video {path.name}")
        try:
            width = _positive_integer(capture.get(cv2.CAP_PROP_FRAME_WIDTH), "video width")
            height = _positive_integer(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), "video height")
            frame_count = _positive_integer(
                capture.get(cv2.CAP_PROP_FRAME_COUNT), "video frame count"
            )
            fps = _positive_float(capture.get(cv2.CAP_PROP_FPS), "video FPS")
        except Exception:
            capture.release()
            raise
        self._path = path
        self._cv2 = cv2
        self._capture = capture
        self._facts = VideoFacts(width, height, fps, frame_count)
        self._closed = False

    @property
    def facts(self) -> VideoFacts:
        return self._facts

    def samples(self, cadence_seconds: float) -> Iterator[VideoSample]:
        if self._closed:
            raise ReviewPackageError("video reader is closed")
        step = max(1, math.floor(self._facts.fps * cadence_seconds))
        if step / self._facts.fps > _MAX_CADENCE_SECONDS:
            raise ReviewPackageError("video FPS cannot satisfy the maximum one-second cadence")
        sampled = 0
        frame_index = 0
        while frame_index < self._facts.frame_count:
            if not self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
                raise ReviewPackageError(f"could not seek frame {frame_index} in {self._path.name}")
            ok, image = self._capture.read()
            if not ok or image is None:
                raise ReviewPackageError(
                    f"could not decode sampled frame {frame_index} in {self._path.name}"
                )
            shape = getattr(image, "shape", None)
            if shape != (self._facts.height, self._facts.width, 3):
                raise ReviewPackageError("decoded frame shape changed during review preparation")
            sampled += 1
            if sampled > _MAX_SAMPLES:
                raise ReviewPackageError(f"video {self._path.name} exceeds the sample limit")
            yield VideoSample(
                frame_index=frame_index,
                video_time_seconds=frame_index / self._facts.fps,
                payload=bytes(image.tobytes()),
                drawable=image.copy(),
            )
            frame_index += step

    def write_jpeg(self, path: Path, drawable: object) -> None:
        if not self._cv2.imwrite(str(path), drawable, [self._cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ReviewPackageError("OpenCV could not write sampled JPEG")

    def write_overlay(
        self, path: Path, drawable: object, proposal: DraftFrameProposal
    ) -> object:
        image = drawable.copy()
        height, width = image.shape[:2]
        for detection in proposal.detections:
            box = detection.bounding_box
            x1, y1 = int(box.x1 * width), int(box.y1 * height)
            x2, y2 = int(box.x2 * width), int(box.y2 * height)
            color = (0, 165, 255) if detection.class_name.startswith("NO-") else (0, 200, 0)
            self._cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            self._cv2.putText(
                image,
                label,
                (x1, max(14, y1 - 5)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                self._cv2.LINE_AA,
            )
        for person in proposal.persons:
            box = person.bounding_box
            x1, y1 = int(box.x1 * width), int(box.y1 * height)
            self._cv2.putText(
                image,
                f"draft-track {person.provisional_track_id}",
                (x1, min(height - 4, y1 + 14)),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 128, 0),
                1,
                self._cv2.LINE_AA,
            )
        banner = f"DRAFT / HUMAN REVIEW REQUIRED / {proposal.review_priority}"
        self._cv2.rectangle(image, (0, 0), (min(width, 620), 24), (32, 32, 32), -1)
        self._cv2.putText(
            image,
            banner,
            (5, 17),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            self._cv2.LINE_AA,
        )
        if not self._cv2.imwrite(str(path), image, [self._cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ReviewPackageError("OpenCV could not write overlay JPEG")
        return image

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()


DetectorLoader = Callable[[Path, RunnerFactory], tuple[Any, Any, Any, Mapping[int, str]]]


@dataclass(frozen=True, slots=True)
class ReviewPackageServices:
    runtime_context: Callable[[], AbstractContextManager[object]]
    runner_factory: RunnerFactory
    detector_loader: DetectorLoader
    video_factory: Callable[[Path], ReviewVideo]
    git_facts: Callable[[Path], Mapping[str, object]]


def _default_detector_loader(path: Path, factory: RunnerFactory) -> tuple[Any, Any, Any, Any]:
    return load_yolo11_detector(path, runner_factory=factory)


def _default_services() -> ReviewPackageServices:
    from smartsite_ai.benchmark.runtime import git_facts

    return ReviewPackageServices(
        runtime_context=ultralytics_runtime_sandbox,
        runner_factory=UltralyticsYoloRunner,
        detector_loader=_default_detector_loader,
        video_factory=OpenCvReviewVideo,
        git_facts=git_facts,
    )


def _cadence(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cadence must be numeric") from error
    if not math.isfinite(number) or not _MIN_CADENCE_SECONDS <= number <= _MAX_CADENCE_SECONDS:
        raise argparse.ArgumentTypeError("cadence must be finite and between 0.1 and 1.0 seconds")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartsite-ai-prepare-video-review",
        description="Generate a local DRAFT PPE pre-label package; human review remains mandatory.",
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--artifact-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cadence-seconds", type=_cadence, default=0.5)
    return parser


def _regular_input(
    path: Path, *, label: str, suffixes: frozenset[str], max_bytes: int = _MAX_VIDEO_BYTES
) -> Path:
    if not path.is_absolute():
        raise ReviewPackageError(f"{label} path must be absolute")
    _reject_link_components(path, label=label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReviewPackageError(f"{label} does not exist") from error
    if is_link_like(path) or not resolved.is_file() or resolved.suffix.casefold() not in suffixes:
        raise ReviewPackageError(f"{label} must be a supported regular local file")
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > max_bytes:
        raise ReviewPackageError(f"{label} size is outside supported bounds")
    return resolved


def _new_output(path: Path, *, repository_root: Path) -> Path:
    if not path.is_absolute():
        raise ReviewPackageError("output directory must be absolute")
    output = path.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise ReviewPackageError("output directory must not already exist")
    _reject_link_components(output.parent, label="output parent")
    if not output.parent.is_dir() or is_link_like(output.parent):
        raise ReviewPackageError("output parent must be an existing regular directory")
    try:
        relative = output.relative_to(repository_root)
    except ValueError:
        return output
    result = subprocess.run(
        ["git", "-C", str(repository_root), "check-ignore", "--quiet", "--", str(relative)],
        check=False,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise ReviewPackageError("in-repository output directory must be ignored by Git")
    return output


def _reject_link_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    try:
        for component in path.absolute().parts[1:]:
            current /= component
            if current.exists() and is_link_like(current):
                raise ReviewPackageError(f"{label} must not traverse links or junctions")
    except OSError as error:
        raise ReviewPackageError(f"{label} path boundary cannot be inspected") from error


def _stable_sha256(path: Path) -> str:
    """Hash one regular file and reject replacement or mutation during the read."""

    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise ReviewPackageError(f"could not hash local input {path.name}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ReviewPackageError(f"local input changed while hashing: {path.name}")
    return digest.hexdigest()


def _verified_git(facts: Mapping[str, object]) -> dict[str, object]:
    commit = facts.get("commitSha")
    if (
        facts.get("available") is not True
        or facts.get("dirty") is not False
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ReviewPackageError("review package generation requires a clean Git commit")
    return dict(facts)


def _box(value: Any) -> dict[str, float]:
    return {"x1": value.x1, "y1": value.y1, "x2": value.x2, "y2": value.y2}


def _full_frame_region(camera_id: str) -> Any:
    configuration = CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": camera_id,
            "regions": (
                {
                    "regionId": _REGION_ID,
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
                },
            ),
        }
    )
    return configuration.regions[0]


def _validate_video_facts(facts: VideoFacts) -> None:
    if not 1 <= facts.width <= 16_384 or not 1 <= facts.height <= 16_384:
        raise ReviewPackageError("video dimensions exceed detector contract bounds")
    if not 1 <= facts.frame_count <= 10_000_000:
        raise ReviewPackageError("video frame count exceeds review package bounds")
    if not math.isfinite(facts.fps) or not 0.0 < facts.fps <= 1_000.0:
        raise ReviewPackageError("video FPS exceeds review package bounds")
    if not math.isfinite(facts.duration_seconds) or not 0.0 < facts.duration_seconds <= 604_800:
        raise ReviewPackageError("video duration exceeds review package bounds")


def _review_flags(batch: Any, tracked: Any, associations: Sequence[Any]) -> tuple[str, ...]:
    flags = {"PROVISIONAL_TRACK_IDS"}
    classes = {item.class_name.casefold() for item in batch.detections}
    if not tracked.persons:
        flags.add("NO_PERSON_DETECTED")
    if len(tracked.persons) > 1:
        flags.add("MULTIPLE_PEOPLE")
    if any(item.confidence < 0.5 for item in batch.detections):
        flags.add("LOW_CONFIDENCE_DETECTION")
    if any(item.status == "MISSING" for item in associations):
        flags.add("NEGATIVE_PPE_EVIDENCE")
    for present, missing in (("hardhat", "no-hardhat"), ("safety vest", "no-safety vest")):
        if present in classes and missing in classes:
            flags.add("CONFLICTING_PPE_EVIDENCE")
    ppe_detections = sum(
        item.class_name.casefold() != "person" for item in batch.detections
    )
    if ppe_detections > len(associations):
        flags.add("UNASSOCIATED_PPE_DETECTION")
    return tuple(sorted(flags))


def _priority(flags: Sequence[str]) -> str:
    if any(
        flag in flags
        for flag in (
            "NEGATIVE_PPE_EVIDENCE",
            "CONFLICTING_PPE_EVIDENCE",
            "UNASSOCIATED_PPE_DETECTION",
        )
    ):
        return "HIGH"
    if "LOW_CONFIDENCE_DETECTION" in flags or "NO_PERSON_DETECTED" in flags:
        return "MEDIUM"
    return "STANDARD"


async def _proposal(
    *,
    detector: Any,
    tracker: IoUPersonTracker,
    pipeline: PpePipeline,
    sample: VideoSample,
    facts: VideoFacts,
    clip_id: str,
    session_id: Any,
) -> DraftFrameProposal:
    frame = FrameEnvelope(
        stream_id=f"review-{clip_id}",
        session_id=session_id,
        camera_external_id=f"review-{clip_id}",
        captured_at=datetime.fromtimestamp(sample.video_time_seconds, tz=UTC),
        width=facts.width,
        height=facts.height,
        sequence_number=sample.frame_index,
        pixel_format="BGR24",
        payload=sample.payload,
    )
    batch = await detector.detect(frame)
    tracked = tracker.update(batch)
    associations = pipeline.process(tracked, _full_frame_region(frame.camera_external_id))
    flags = _review_flags(batch, tracked, associations)
    stem = f"frame-{sample.frame_index:010d}"
    return DraftFrameProposal.model_validate(
        {
            "schemaVersion": "1.0.0",
            "status": "DRAFT",
            "clipId": clip_id,
            "frameIndex": sample.frame_index,
            "videoTimeSeconds": sample.video_time_seconds,
            "width": facts.width,
            "height": facts.height,
            "imagePath": str(PurePosixPath("frames", clip_id, f"{stem}.jpg")),
            "overlayPath": str(PurePosixPath("overlays", clip_id, f"{stem}.jpg")),
            "detections": tuple(
                ProposalDetection.model_validate(
                    {
                        "proposalId": f"d{index:04d}",
                        "classId": detection.class_id,
                        "className": detection.class_name,
                        "confidence": detection.confidence,
                        "boundingBox": _box(detection.bounding_box),
                    }
                )
                for index, detection in enumerate(batch.detections)
            ),
            "persons": tuple(
                ProposalPerson.model_validate(
                    {
                        "provisionalTrackId": person.track_id,
                        "confidence": person.detection.confidence,
                        "boundingBox": _box(person.bounding_box),
                    }
                )
                for person in tracked.persons
            ),
            "ppeAssociations": tuple(
                ProposalPpeAssociation.model_validate(
                    {
                        "provisionalTrackId": item.track_id,
                        "ppeItem": item.ppe_item,
                        "proposedStatus": item.status,
                        "confidence": item.confidence,
                        "boundingBox": _box(item.bounding_box),
                    }
                )
                for item in associations
                if item.confidence is not None and item.bounding_box is not None
            ),
            "reviewPriority": _priority(flags),
            "reviewFlags": flags,
        }
    )


def _atomic_text(path: Path, text: str) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _publish_package(
    *,
    output: Path,
    videos: Sequence[Path],
    video_hashes: Sequence[str],
    artifact_spec: Path,
    cadence_seconds: float,
    services: ReviewPackageServices,
    repository_root: Path,
) -> None:
    git_start = _verified_git(services.git_facts(repository_root))
    artifact_spec_hash = _stable_sha256(artifact_spec)
    working = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    runner = None
    rows: list[str] = []
    clip_manifests: list[dict[str, object]] = []
    total_samples = 0
    try:
        (working / "frames").mkdir()
        (working / "overlays").mkdir()
        with services.runtime_context():
            detector, runner, artifact, _class_map = services.detector_loader(
                artifact_spec, services.runner_factory
            )
            try:
                _reject_link_components(artifact.artifact_path, label="model checkpoint")
                if artifact.model_family != "yolo11s":
                    raise ReviewPackageError("review package requires modelFamily yolo11s")
                for clip_number, (video, media_hash) in enumerate(
                    zip(videos, video_hashes, strict=True), start=1
                ):
                    clip_id = f"clip-{clip_number:03d}-{media_hash[:12]}"
                    (working / "frames" / clip_id).mkdir()
                    (working / "overlays" / clip_id).mkdir()
                    reader = services.video_factory(video)
                    sampled = 0
                    try:
                        facts = reader.facts
                        _validate_video_facts(facts)
                        sample_step = max(1, math.floor(facts.fps * cadence_seconds))
                        expected = math.ceil(facts.frame_count / sample_step)
                        if expected > _MAX_SAMPLES or total_samples + expected > _MAX_SAMPLES:
                            raise ReviewPackageError(
                                "requested package exceeds the total sample limit"
                            )
                        tracker = IoUPersonTracker()
                        pipeline = PpePipeline()
                        session_id = uuid5(
                            NAMESPACE_URL, f"smartsite-draft-review:{media_hash}"
                        )
                        previous_time: float | None = None
                        for sample in reader.samples(cadence_seconds):
                            gap = (
                                sample.video_time_seconds - previous_time
                                if previous_time is not None
                                else 0.0
                            )
                            if gap > 1.0 + 1e-9:
                                raise ReviewPackageError(
                                    "decoded sample cadence exceeds one second"
                                )
                            previous_time = sample.video_time_seconds
                            proposal = asyncio.run(
                                _proposal(
                                    detector=detector,
                                    tracker=tracker,
                                    pipeline=pipeline,
                                    sample=sample,
                                    facts=facts,
                                    clip_id=clip_id,
                                    session_id=session_id,
                                )
                            )
                            image_path = working / Path(proposal.image_path)
                            overlay_path = working / Path(proposal.overlay_path)
                            reader.write_jpeg(image_path, sample.drawable)
                            reader.write_overlay(overlay_path, sample.drawable, proposal)
                            rows.append(
                                json.dumps(
                                    proposal.model_dump(mode="json", by_alias=True),
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                            sampled += 1
                            total_samples += 1
                        if sampled == 0:
                            raise ReviewPackageError("video produced no review samples")
                        clip_manifests.append(
                            {
                                "clipId": clip_id,
                                "mediaPath": str(video),
                                "sha256Before": media_hash,
                                "sha256After": media_hash,
                                "sizeBytes": video.stat().st_size,
                                "width": facts.width,
                                "height": facts.height,
                                "fps": facts.fps,
                                "frameCount": facts.frame_count,
                                "durationSeconds": facts.duration_seconds,
                                "sampleCount": sampled,
                            }
                        )
                    finally:
                        reader.close()
            finally:
                active_runner = runner
                runner = None
                active_runner.close()
        checkpoint_after = _stable_sha256(artifact.resolved_path)
        spec_after = _stable_sha256(artifact_spec)
        if checkpoint_after != artifact.actual_sha256 or spec_after != artifact_spec_hash:
            raise ReviewPackageError("model artifact changed during review preparation")
        for item, video in zip(clip_manifests, videos, strict=True):
            media_after = _stable_sha256(video)
            item["sha256After"] = media_after
            if media_after != item["sha256Before"]:
                raise ReviewPackageError("source video changed during review preparation")
        git_finish = _verified_git(services.git_facts(repository_root))
        if git_finish != git_start:
            raise ReviewPackageError("Git provenance changed during review preparation")
        _atomic_text(working / "proposals.jsonl", "\n".join(rows) + "\n")
        manifest = DraftReviewPackageManifest.model_validate({
            "schemaVersion": "1.0.0",
            "status": "DRAFT",
            "packageType": "PPE_TEMPORAL_PRELABEL_REVIEW_PACKAGE",
            "groundTruth": False,
            "humanReviewRequired": True,
            "warning": "Machine proposals only. This package is not reviewed ground truth.",
            "generatedAtUtc": datetime.now(UTC),
            "configuration": {
                "cadenceSeconds": cadence_seconds,
                "maximumAllowedGapSeconds": 1.0,
                "trackerScope": "reset-at-each-clip-boundary",
                "trackIdSemantics": "provisional-clip-local-not-worker-identity",
                "localOnly": True,
            },
            "artifact": {
                "artifactSpecPath": str(artifact_spec),
                "artifactSpecSha256Before": artifact_spec_hash,
                "artifactSpecSha256After": spec_after,
                "checkpointPath": str(artifact.resolved_path),
                "checkpointSha256Before": artifact.actual_sha256,
                "checkpointSha256After": checkpoint_after,
                "artifactId": artifact.artifact_id,
                "version": artifact.version,
                "modelFamily": artifact.model_family,
                "classMap": {str(key): value for key, value in artifact.class_map},
            },
            "clips": tuple(clip_manifests),
            "proposalCount": total_samples,
            "proposalsPath": "proposals.jsonl",
            "git": git_finish,
        })
        _atomic_text(
            working / "manifest.json",
            json.dumps(
                manifest.model_dump(mode="json", by_alias=True),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        working.replace(output)
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(working)
        raise
    finally:
        if runner is not None:
            runner.close()


def run(
    argv: Sequence[str] | None = None,
    *,
    services: ReviewPackageServices | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    try:
        if not 1 <= len(args.input) <= _MAX_VIDEOS:
            raise ReviewPackageError(f"provide between 1 and {_MAX_VIDEOS} videos")
        videos = [
            _regular_input(path, label=f"input video {index}", suffixes=_VIDEO_SUFFIXES)
            for index, path in enumerate(args.input, start=1)
        ]
        if len(set(videos)) != len(videos):
            raise ReviewPackageError("video paths must be distinct after resolution")
        video_hashes = [_stable_sha256(video) for video in videos]
        if len(set(video_hashes)) != len(video_hashes):
            raise ReviewPackageError("video contents must be distinct")
        artifact_spec = _regular_input(
            args.artifact_spec,
            label="artifact spec",
            suffixes=frozenset({".json"}),
            max_bytes=256 * 1024,
        )
        output = _new_output(args.output_dir, repository_root=repository_root)
        _publish_package(
            output=output,
            videos=videos,
            video_hashes=video_hashes,
            artifact_spec=artifact_spec,
            cadence_seconds=args.cadence_seconds,
            services=services or _default_services(),
            repository_root=repository_root,
        )
    except KeyboardInterrupt:
        print("review package interrupted; no package was published", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"review package failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"DRAFT review package created: {output}")
    return 0


def _positive_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReviewPackageError(f"{label} is invalid") from error
    if not math.isfinite(number) or number <= 0:
        raise ReviewPackageError(f"{label} is invalid")
    return number


def _positive_integer(value: object, label: str) -> int:
    number = _positive_float(value, label)
    if not number.is_integer():
        raise ReviewPackageError(f"{label} is invalid")
    return int(number)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()


__all__ = [
    "OpenCvReviewVideo",
    "ReviewPackageError",
    "ReviewPackageServices",
    "VideoFacts",
    "VideoSample",
    "build_parser",
    "main",
    "run",
]
