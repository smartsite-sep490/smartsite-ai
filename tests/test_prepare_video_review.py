import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.review_package import DraftFrameProposal
from smartsite_ai.evaluation.video_corpus import ReviewedCorpusSource, ReviewedFrame
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.tools.prepare_video_review import (
    OpenCvReviewVideo,
    ReviewPackageError,
    ReviewPackageServices,
    VideoFacts,
    VideoSample,
    _new_output,
    _review_flags,
    _sample_frame_indexes,
    run,
)


def detection(
    class_id: int, class_name: str, bounds: tuple[float, float, float, float], confidence: float
) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(x1=bounds[0], y1=bounds[1], x2=bounds[2], y2=bounds[3]),
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


class MalformedVideo(FakeVideo):
    def __init__(self, source: Path, samples: tuple[VideoSample, ...]) -> None:
        super().__init__(source)
        self._samples = samples

    def samples(self, _cadence: float):
        yield from self._samples


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
    publisher: object | None = None,
    git_facts: object | None = None,
    video_factory: object | None = None,
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
        video_factory=video_factory
        or (lambda path: FakeVideo(path, mutate_on_close=mutate_on_close)),
        git_facts=git_facts
        or (
            lambda _root: {
                "available": True,
                "commitSha": "1" * 40,
                "dirty": False,
            }
        ),
        publisher=publisher or (lambda source, target: source.rename(target)),
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


def test_sample_indexes_include_first_and_last_frame_exactly_once() -> None:
    facts = VideoFacts(width=640, height=480, fps=30.0, frame_count=60)

    assert _sample_frame_indexes(facts, 1.0) == (0, 30, 59)


class TinyImage:
    shape = (2, 3, 3)

    def tobytes(self) -> bytes:
        return b"\x00" * 18

    def copy(self) -> "TinyImage":
        return self


class FakeCapture:
    def __init__(
        self,
        *,
        seek_offset: float = 0.0,
        timestamps_ms: dict[int, float] | None = None,
        metadata_error: BaseException | None = None,
        open_error: BaseException | None = None,
    ) -> None:
        self.seek_offset = seek_offset
        self.timestamps_ms = timestamps_ms or {0: 0.0, 30: 1_000.0, 59: 1_966.0}
        self.metadata_error = metadata_error
        self.open_error = open_error
        self.position = 0.0
        self.last_read_index = 0
        self.released = False

    def isOpened(self) -> bool:
        if self.open_error is not None:
            raise self.open_error
        return True

    def get(self, prop: int) -> float:
        if self.metadata_error is not None and prop == FakeCv2.CAP_PROP_FRAME_WIDTH:
            raise self.metadata_error
        if prop == FakeCv2.CAP_PROP_FRAME_WIDTH:
            return 3.0
        if prop == FakeCv2.CAP_PROP_FRAME_HEIGHT:
            return 2.0
        if prop == FakeCv2.CAP_PROP_FRAME_COUNT:
            return 60.0
        if prop == FakeCv2.CAP_PROP_FPS:
            return 30.0
        if prop == FakeCv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop == FakeCv2.CAP_PROP_POS_MSEC:
            return self.timestamps_ms[self.last_read_index]
        raise AssertionError(f"unexpected property {prop}")

    def set(self, prop: int, value: float) -> bool:
        assert prop == FakeCv2.CAP_PROP_POS_FRAMES
        self.position = value + self.seek_offset
        return True

    def read(self) -> tuple[bool, TinyImage]:
        self.last_read_index = int(round(self.position))
        self.position += 1.0
        return True, TinyImage()

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_POS_MSEC = 0
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _path: str) -> FakeCapture:
        return self.capture


def test_opencv_reader_uses_verified_positions_and_real_timestamps(tmp_path: Path) -> None:
    capture = FakeCapture()
    reader = OpenCvReviewVideo(tmp_path / "video.mp4", cv2_module=FakeCv2(capture))

    samples = list(reader.samples(1.0))

    assert [sample.frame_index for sample in samples] == [0, 30, 59]
    assert [sample.video_time_seconds for sample in samples] == [0.0, 1.0, 1.966]


def test_opencv_reader_rejects_inaccurate_seek(tmp_path: Path) -> None:
    reader = OpenCvReviewVideo(
        tmp_path / "video.mp4", cv2_module=FakeCv2(FakeCapture(seek_offset=1.0))
    )

    with pytest.raises(ReviewPackageError, match="did not select requested frame"):
        list(reader.samples(1.0))


@pytest.mark.parametrize(
    "timestamps",
    [
        {0: 0.0, 30: float("nan"), 59: 1_966.0},
        {0: 0.0, 30: 0.0, 59: 1_966.0},
    ],
)
def test_opencv_reader_rejects_invalid_or_nonmonotonic_timestamp(
    tmp_path: Path, timestamps: dict[int, float]
) -> None:
    reader = OpenCvReviewVideo(
        tmp_path / "video.mp4", cv2_module=FakeCv2(FakeCapture(timestamps_ms=timestamps))
    )

    with pytest.raises(ReviewPackageError, match="timestamp"):
        list(reader.samples(1.0))


def test_opencv_reader_releases_capture_when_metadata_read_is_interrupted(tmp_path: Path) -> None:
    capture = FakeCapture(metadata_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        OpenCvReviewVideo(tmp_path / "video.mp4", cv2_module=FakeCv2(capture))

    assert capture.released is True


def test_opencv_reader_releases_capture_when_open_check_is_interrupted(tmp_path: Path) -> None:
    capture = FakeCapture(open_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        OpenCvReviewVideo(tmp_path / "video.mp4", cv2_module=FakeCv2(capture))

    assert capture.released is True


def test_generated_ids_are_package_unique_and_associations_reference_detections(
    tmp_path: Path,
) -> None:
    videos, spec, checkpoint = inputs(tmp_path, second=True)
    output = tmp_path / "review-draft"
    injected, _runner = services(checkpoint)

    assert run(command(videos, spec, output), services=injected) == 0
    rows = [json.loads(line) for line in (output / "proposals.jsonl").read_text().splitlines()]
    frame_ids = [row["frameProposalId"] for row in rows]
    detection_ids = [item["proposalId"] for row in rows for item in row["detections"]]
    assert len(frame_ids) == len(set(frame_ids))
    assert len(detection_ids) == len(set(detection_ids))
    for row in rows:
        row_detection_ids = {item["proposalId"] for item in row["detections"]}
        assert all(
            item["sourceDetectionProposalId"] in row_detection_ids
            for item in row["ppeAssociations"]
        )


def _valid_frame_proposal() -> dict[str, object]:
    box = {"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}
    return {
        "schemaVersion": "1.0.0",
        "status": "DRAFT",
        "frameProposalId": "clip-001:frame-0000000000",
        "clipId": "clip-001",
        "frameIndex": 0,
        "videoTimeSeconds": 0.0,
        "width": 640,
        "height": 480,
        "imagePath": "frames/clip-001/frame.jpg",
        "overlayPath": "overlays/clip-001/frame.jpg",
        "detections": [
            {
                "proposalId": "clip-001:frame-0000000000:detection-0000",
                "classId": 1,
                "className": "Hardhat",
                "confidence": 0.9,
                "boundingBox": box,
            }
        ],
        "persons": [{"provisionalTrackId": 1, "confidence": 0.95, "boundingBox": box}],
        "ppeAssociations": [
            {
                "provisionalTrackId": 1,
                "sourceDetectionProposalId": "clip-001:frame-0000000000:detection-0000",
                "ppeItem": "HARD_HAT",
                "proposedStatus": "PRESENT",
                "confidence": 0.9,
                "boundingBox": box,
            }
        ],
        "reviewPriority": "STANDARD",
        "reviewFlags": ["PROVISIONAL_TRACK_IDS"],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["detections"].append(value["detections"][0].copy()),
        lambda value: value["persons"].append(value["persons"][0].copy()),
        lambda value: value["ppeAssociations"][0].update({"sourceDetectionProposalId": "missing"}),
        lambda value: value["ppeAssociations"][0].update({"provisionalTrackId": 2}),
        lambda value: value["ppeAssociations"][0].update({"proposedStatus": "MISSING"}),
        lambda value: value["ppeAssociations"][0].update({"confidence": 0.8}),
        lambda value: value["ppeAssociations"][0].update(
            {"boundingBox": {"x1": 0.15, "y1": 0.1, "x2": 0.3, "y2": 0.3}}
        ),
        lambda value: value["ppeAssociations"].append(value["ppeAssociations"][0].copy()),
    ],
)
def test_frame_contract_rejects_duplicate_or_broken_references(mutation: object) -> None:
    value = _valid_frame_proposal()
    mutation(value)

    with pytest.raises(ValidationError):
        DraftFrameProposal.model_validate(value)


def test_mixed_classes_for_two_people_are_not_reported_as_conflicting() -> None:
    batch = SimpleNamespace(
        detections=(
            SimpleNamespace(class_name="Hardhat", confidence=0.9),
            SimpleNamespace(class_name="NO-Hardhat", confidence=0.9),
        )
    )
    tracked = SimpleNamespace(persons=(SimpleNamespace(track_id=1), SimpleNamespace(track_id=2)))
    associations = (
        SimpleNamespace(track_id=1, ppe_item="HARD_HAT", status="PRESENT"),
        SimpleNamespace(track_id=2, ppe_item="HARD_HAT", status="MISSING"),
    )

    flags = _review_flags(batch, tracked, associations)

    assert "CONFLICTING_PPE_EVIDENCE" not in flags
    assert "MIXED_PPE_CLASSES_IN_FRAME" in flags
    assert "PPE_STATUS_UNKNOWN" in flags


def test_unknown_ppe_is_not_inferred_as_missing() -> None:
    batch = SimpleNamespace(detections=(SimpleNamespace(class_name="Person", confidence=0.95),))
    tracked = SimpleNamespace(persons=(SimpleNamespace(track_id=1),))

    flags = _review_flags(batch, tracked, ())

    assert "PPE_STATUS_UNKNOWN" in flags
    assert "NEGATIVE_PPE_EVIDENCE" not in flags


def test_output_rejects_a_linked_parent_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ReviewPackageError, match="must not traverse"):
        _new_output(linked_parent / "review-draft", repository_root=tmp_path)


def test_final_git_recheck_rejects_provenance_change(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    calls = 0

    def changing_git(_root: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "available": True,
            "commitSha": ("1" if calls < 3 else "2") * 40,
            "dirty": False,
        }

    injected, runner = services(checkpoint, git_facts=changing_git)

    assert run(command(videos, spec, output), services=injected) == 1
    assert calls == 3
    assert not output.exists()
    assert runner.closed is True


@pytest.mark.parametrize("mutated_input", ["spec", "checkpoint", "video"])
def test_final_hash_recheck_rejects_input_mutation(tmp_path: Path, mutated_input: str) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    targets = {"spec": spec, "checkpoint": checkpoint, "video": videos[0]}
    calls = 0

    def mutating_git(_root: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            target = targets[mutated_input]
            target.write_bytes(target.read_bytes() + b"changed-after-staging")
        return {
            "available": True,
            "commitSha": "1" * 40,
            "dirty": False,
        }

    injected, runner = services(checkpoint, git_facts=mutating_git)

    assert run(command(videos, spec, output), services=injected) == 1
    assert not output.exists()
    assert runner.closed is True


@pytest.mark.parametrize(
    "samples",
    [
        (
            VideoSample(0, 0.0, b"\x00" * 48, object()),
            VideoSample(0, 0.5, b"\x01" * 48, object()),
        ),
        (
            VideoSample(0, 0.5, b"\x00" * 48, object()),
            VideoSample(1, 0.4, b"\x01" * 48, object()),
        ),
        (
            VideoSample(0, 0.0, b"\x00" * 48, object()),
            VideoSample(1, float("nan"), b"\x01" * 48, object()),
        ),
    ],
)
def test_package_boundary_rejects_malformed_frame_sequence(
    tmp_path: Path, samples: tuple[VideoSample, ...]
) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, runner = services(
        checkpoint,
        video_factory=lambda path: MalformedVideo(path, samples),
    )

    assert run(command(videos, spec, output), services=injected) == 1
    assert not output.exists()
    assert runner.closed is True


def test_interrupt_after_owned_rename_removes_only_owned_output(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"

    def rename_then_interrupt(source: Path, target: Path) -> None:
        source.rename(target)
        raise KeyboardInterrupt

    injected, runner = services(checkpoint, publisher=rename_then_interrupt)

    assert run(command(videos, spec, output), services=injected) == 130
    assert not output.exists()
    assert runner.closed is True


def test_publication_race_never_deletes_an_external_output(tmp_path: Path) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"

    def racing_publisher(_source: Path, target: Path) -> None:
        target.mkdir()
        (target / "external.txt").write_text("keep", encoding="utf-8")
        raise KeyboardInterrupt

    injected, _runner = services(checkpoint, publisher=racing_publisher)

    assert run(command(videos, spec, output), services=injected) == 130
    assert (output / "external.txt").read_text(encoding="utf-8") == "keep"


def test_successful_publication_retains_ownership_marker_for_interrupt_safety(
    tmp_path: Path,
) -> None:
    videos, spec, checkpoint = inputs(tmp_path)
    output = tmp_path / "review-draft"
    injected, _runner = services(checkpoint)

    assert run(command(videos, spec, output), services=injected) == 0

    marker = output / ".smartsite-publication-owner"
    assert marker.is_file()
    assert len(marker.read_text(encoding="ascii")) == 64
