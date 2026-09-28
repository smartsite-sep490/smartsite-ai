import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.video_corpus import ReviewedCorpusSource, ReviewedFrame
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.tools.prepare_video_review import (
    ReviewPackageServices,
    VideoFacts,
    VideoSample,
    run,
)


def detection(
    class_id: int, class_name: str, bounds: tuple[float, float, float, float], confidence: float
) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(
            x1=bounds[0], y1=bounds[1], x2=bounds[2], y2=bounds[3]
        ),
    )


class FakeDetector:
    async def detect(self, frame: Any) -> DetectionBatch:
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="smartsite-yolo11s-ppe",
            model_version="test-v1",
            model_sha256="a" * 64,
            detections=(
                detection(0, "Person", (0.1, 0.1, 0.9, 0.9), 0.95),
                detection(2, "NO-Hardhat", (0.3, 0.15, 0.5, 0.3), 0.88),
            ),
        )


class FailingDetector:
    async def detect(self, _frame: Any) -> DetectionBatch:
        raise RuntimeError("synthetic detector failure")


class FakeRunner:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeVideo:
    def __init__(self, source: Path, *, mutate_on_close: bool = False) -> None:
        self._source = source
        self._mutate_on_close = mutate_on_close
        self._facts = VideoFacts(width=4, height=4, fps=2.0, frame_count=2)

    @property
    def facts(self) -> VideoFacts:
        return self._facts

    def samples(self, _cadence: float):
        yield VideoSample(0, 0.0, b"\x00" * 48, object())
        yield VideoSample(1, 0.5, b"\x01" * 48, object())

    def write_jpeg(self, path: Path, _drawable: object) -> None:
        path.write_bytes(b"jpeg-original")

    def write_overlay(self, path: Path, _drawable: object, _proposal: object) -> object:
        path.write_bytes(b"jpeg-overlay")
        return object()

    def close(self) -> None:
        if self._mutate_on_close:
            self._source.write_bytes(self._source.read_bytes() + b"changed")


def inputs(tmp_path: Path, *, second: bool = False) -> tuple[list[Path], Path, Path]:
    first = tmp_path / "one.mp4"
    first.write_bytes(b"video-one")
    videos = [first]
    if second:
        other = tmp_path / "two.mp4"
        other.write_bytes(b"video-two")
        videos.append(other)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")
    spec = tmp_path / "artifact.json"
    spec.write_text("{}", encoding="utf-8")
    return videos, spec, checkpoint


def services(
    checkpoint: Path,
    *,
    detector: object | None = None,
    mutate_on_close: bool = False,
) -> tuple[ReviewPackageServices, FakeRunner]:
    runner = FakeRunner()
    artifact = SimpleNamespace(
        model_family="yolo11s",
        artifact_path=checkpoint,
        resolved_path=checkpoint,
        actual_sha256=__import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        artifact_id="smartsite-yolo11s-ppe",
        version="test-v1",
        class_map=(
            (0, "Person"),
            (1, "Hardhat"),
            (2, "NO-Hardhat"),
            (3, "Safety Vest"),
            (4, "NO-Safety Vest"),
        ),
    )

    def load(_path: Path, _factory: object):
        return detector or FakeDetector(), runner, artifact, dict(artifact.class_map)

    values = ReviewPackageServices(
        runtime_context=nullcontext,
        runner_factory=lambda: runner,
        detector_loader=load,
        video_factory=lambda path: FakeVideo(path, mutate_on_close=mutate_on_close),
        git_facts=lambda _root: {
            "available": True,
            "commitSha": "1" * 40,
            "dirty": False,
        },
    )
    return values, runner


def command(videos: list[Path], spec: Path, output: Path) -> list[str]:
    result: list[str] = []
    for video in videos:
        result.extend(("--input", str(video.resolve())))
    result.extend(
        (
            "--artifact-spec",
            str(spec.resolve()),
            "--output-dir",
            str(output.resolve()),
            "--cadence-seconds",
            "0.5",
        )
    )
    return result


def test_builds_atomic_draft_package_and_resets_tracks_per_clip(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path, second=True)
    output = tmp_path / "review-draft"
    injected, runner = services(checkpoint)

    assert run(command(videos, spec, output), services=injected) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "proposals.jsonl").read_text().splitlines()]
    assert manifest["status"] == "DRAFT"
    assert manifest["groundTruth"] is False
    assert manifest["humanReviewRequired"] is True
    assert manifest["proposalCount"] == 4
    assert [row["persons"][0]["provisionalTrackId"] for row in (rows[0], rows[2])] == [1, 1]
    assert all(row["status"] == "DRAFT" for row in rows)
    assert all(row["ppeAssociations"][0]["proposedStatus"] == "MISSING" for row in rows)
    assert all("PROVISIONAL_TRACK_IDS" in row["reviewFlags"] for row in rows)
    assert len(list((output / "frames").rglob("*.jpg"))) == 4
    assert len(list((output / "overlays").rglob("*.jpg"))) == 4
    assert runner.closed is True


def test_draft_outputs_cannot_validate_as_reviewed_corpus_inputs(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, _runner = services(checkpoint)
    assert run(command(videos, spec, output), services=injected) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    proposal = json.loads((output / "proposals.jsonl").read_text().splitlines()[0])

    with pytest.raises(ValidationError):
        ReviewedCorpusSource.model_validate(manifest)
    with pytest.raises(ValidationError):
        ReviewedFrame.model_validate(proposal)
    serialized = json.dumps({"manifest": manifest, "proposal": proposal})
    assert "reviewedBy" not in serialized
    assert "reviewedAt" not in serialized
    assert "episodesPath" not in serialized


def test_rejects_duplicate_video_content_before_creating_output(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path, second=True)
    videos[1].write_bytes(videos[0].read_bytes())
    output = tmp_path / "review-draft"
    injected, _runner = services(checkpoint)

    assert run(command(videos, spec, output), services=injected) == 1
    assert not output.exists()


def test_rejects_output_overwrite_before_loading_detector(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    injected, runner = services(checkpoint)

    assert run(command(videos, spec, output), services=injected) == 1
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert runner.closed is False


def test_failure_cleans_staging_and_closes_runner(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, runner = services(checkpoint, detector=FailingDetector())

    assert run(command(videos, spec, output), services=injected) == 1
    assert not output.exists()
    assert runner.closed is True
    assert not list(tmp_path.glob(".review-draft.*"))


def test_rejects_source_mutation_and_does_not_publish(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, runner = services(checkpoint, mutate_on_close=True)

    assert run(command(videos, spec, output), services=injected) == 1
    assert not output.exists()
    assert runner.closed is True


def test_rejects_sparse_or_invalid_cadence_at_argument_boundary(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, _runner = services(checkpoint)
    args = command(videos, spec, output)
    args[-1] = "1.01"

    with pytest.raises(SystemExit):
        run(args, services=injected)
