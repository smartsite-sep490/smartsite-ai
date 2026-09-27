import hashlib
import json
from pathlib import Path

import pytest

from smartsite_ai.evaluation.dataset import DatasetValidationError, load_evaluation_dataset
from smartsite_ai.evaluation.execution import load_ground_truth_episodes
from smartsite_ai.evaluation.video_corpus import VideoFacts
from smartsite_ai.tools import build_video_evaluation_corpus as corpus_tool
from smartsite_ai.tools.build_video_evaluation_corpus import run


class FakeVideoProbe:
    def __init__(self, facts: VideoFacts | None = None) -> None:
        self.facts = facts or VideoFacts(
            width=640,
            height=480,
            frame_count=300,
            duration_seconds=30.0,
        )

    def probe(self, _path: Path) -> VideoFacts:
        return self.facts

    def frame_times(self, _path: Path, frame_indexes: list[int]) -> dict[int, float]:
        seconds_per_frame = self.facts.duration_seconds / self.facts.frame_count
        return {index: index * seconds_per_frame for index in frame_indexes}


def _person(frame: int, *, person_id: str | None = "person-1") -> dict[str, object]:
    value: dict[str, object] = {
        "annotationId": f"f{frame}-person",
        "className": "Person",
        "boundingBox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
        "visibility": 1.0,
        "observablePpeItems": ["HARD_HAT", "SAFETY_VEST"],
    }
    if person_id is not None:
        value["personInstanceId"] = person_id
    return value


def _ppe(frame: int, class_name: str) -> dict[str, object]:
    return {
        "annotationId": f"f{frame}-{class_name.lower()}",
        "className": class_name,
        "boundingBox": {"x1": 0.2, "y1": 0.1, "x2": 0.35, "y2": 0.25},
        "personInstanceId": "person-1",
        "relatedPersonAnnotationId": f"f{frame}-person",
        "visibility": 1.0,
    }


def _frame(
    index: int,
    time_seconds: float,
    *,
    class_name: str = "NO-Hardhat",
    person_id: str | None = "person-1",
) -> dict[str, object]:
    annotations = [_person(index, person_id=person_id)]
    if class_name:
        annotations.append(_ppe(index, class_name))
    return {
        "clipId": "clip-001",
        "frameIndex": index,
        "videoTimeSeconds": time_seconds,
        "width": 640,
        "height": 480,
        "annotations": annotations,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _inputs(
    tmp_path: Path,
    *,
    frames: list[dict[str, object]] | None = None,
    episodes: list[dict[str, object]] | None = None,
    declared_hash: str | None = None,
) -> tuple[Path, Path]:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"reviewed synthetic video fixture")
    frames_path = tmp_path / "clip.frames.jsonl"
    _write_jsonl(frames_path, frames or [_frame(0, 0.0), _frame(10, 1.0)])
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        episodes
        if episodes is not None
        else [
            {
                "clipId": "clip-001",
                "personInstanceId": "person-1",
                "ppeItem": "HARD_HAT",
                "startTimeSeconds": 0.0,
                "endTimeSeconds": 1.0,
            }
        ],
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "datasetId": "smartsite-reviewed-ppe-video",
                "datasetVersion": "test-v1",
                "sourceUrl": "https://example.com/reviewed-corpus",
                "license": "fixture-only",
                "reviewedBy": "reviewer-1",
                "reviewedAtUtc": "2026-09-28T00:00:00Z",
                "maxFrameGapSeconds": 1.0,
                "episodesPath": str(episodes_path.resolve()),
                "clips": [
                    {
                        "clipId": "clip-001",
                        "mediaPath": str(media.resolve()),
                        "sha256": declared_hash or hashlib.sha256(media.read_bytes()).hexdigest(),
                        "framesPath": str(frames_path.resolve()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, tmp_path / "output"


def _run(source: Path, output: Path, *, facts: VideoFacts | None = None) -> int:
    return run(
        ["--source-manifest", str(source.resolve()), "--output-dir", str(output.resolve())],
        video_probe=FakeVideoProbe(facts),
    )


def test_builds_loader_valid_atomic_corpus_and_rewrites_episode_clip(tmp_path: Path) -> None:
    source, output = _inputs(tmp_path)

    assert _run(source, output) == 0

    loaded = load_evaluation_dataset(output / "evaluation.manifest.json")
    assert len(loaded.get_split("test")) == 2
    assert not loaded.get_split("train")
    assert not loaded.get_split("validation")
    assert loaded.get_split("test")[0].media_path == "media/test/clip-001.mp4"
    episodes = load_ground_truth_episodes(output / "indexes" / "test-episodes.jsonl")
    assert episodes[0].clip_id == "media/test/clip-001.mp4"
    manifest = json.loads((output / "corpus.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["evaluation"]["evaluatedCameraSeconds"] == 1.0
    assert manifest["evaluation"]["rateGateEligible"] is False
    assert manifest["evaluation"]["rateGateIneligibilityReason"] == (
        "requires at least 1800 evaluated camera-seconds"
    )
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_build_is_deterministic_for_same_reviewed_inputs(tmp_path: Path) -> None:
    source, first = _inputs(tmp_path)
    second = tmp_path / "output-2"

    assert _run(source, first) == 0
    assert _run(source, second) == 0

    for relative in (
        "evaluation.manifest.json",
        "corpus.manifest.json",
        "indexes/test.jsonl",
        "indexes/test-episodes.jsonl",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_marks_1800_camera_seconds_rate_gate_eligible(tmp_path: Path) -> None:
    frames = [_frame(second * 10, float(second), class_name="Hardhat") for second in range(1801)]
    episodes: list[dict[str, object]] = []
    source, output = _inputs(tmp_path, frames=frames, episodes=episodes)
    facts = VideoFacts(width=640, height=480, frame_count=18_010, duration_seconds=1801.0)

    assert _run(source, output, facts=facts) == 0
    report = json.loads((output / "corpus.manifest.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["evaluatedCameraSeconds"] == 1800.0
    assert report["evaluation"]["rateGateEligible"] is True
    assert report["evaluation"]["rateGateIneligibilityReason"] is None


def test_rejects_sparse_frames_that_would_inflate_camera_seconds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path, frames=[_frame(0, 0.0), _frame(299, 29.9)])

    assert _run(source, output) == 1
    assert "exceeds maxFrameGapSeconds" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_timestamp_that_does_not_match_decoded_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path, frames=[_frame(0, 0.0), _frame(10, 0.5)])

    assert _run(source, output) == 1
    assert "does not identify the decoded video frame" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([_frame(10, 1.0), _frame(0, 0.0)], "frameIndex values"),
        ([_frame(0, 1.0), _frame(10, 0.5)], "videoTimeSeconds values"),
        ([_frame(300, 1.0), _frame(301, 2.0)], "outside the video"),
        ([_frame(0, 0.0, person_id=None), _frame(10, 1.0)], "personInstanceId"),
    ],
)
def test_rejects_invalid_frame_identity_order_and_bounds_atomically(
    tmp_path: Path,
    frames: list[dict[str, object]],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = _inputs(tmp_path, frames=frames)

    assert _run(source, output) == 1
    assert message in capsys.readouterr().err
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_rejects_orphan_ppe_relationship_atomically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = _frame(0, 0.0)
    invalid["annotations"][1]["relatedPersonAnnotationId"] = "missing-person"
    source, output = _inputs(tmp_path, frames=[invalid, _frame(10, 1.0)])

    assert _run(source, output) == 1
    assert "non-existent relatedPersonAnnotationId" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_positive_and_negative_same_person_ppe_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _frame(0, 0.0)
    positive = _ppe(0, "Hardhat")
    positive["annotationId"] = "f0-hardhat-positive"
    frame["annotations"].append(positive)
    source, output = _inputs(tmp_path, frames=[frame, _frame(10, 1.0)])

    assert _run(source, output) == 1
    assert "conflicting HARD_HAT labels" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_duplicate_person_identity_in_one_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _frame(0, 0.0)
    duplicate = _person(0)
    duplicate["annotationId"] = "f0-person-duplicate"
    frame["annotations"].append(duplicate)
    source, output = _inputs(tmp_path, frames=[frame, _frame(10, 1.0)])

    assert _run(source, output) == 1
    assert "repeats a Person personInstanceId" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_positive_evidence_inside_missing_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(
        tmp_path,
        frames=[_frame(0, 0.0), _frame(10, 1.0, class_name="Hardhat")],
    )

    assert _run(source, output) == 1
    assert "contains reviewed positive-PPE evidence" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("episodes", "message"),
    [
        (
            [
                {
                    "clipId": "clip-001",
                    "personInstanceId": "person-1",
                    "ppeItem": "HARD_HAT",
                    "startTimeSeconds": 0.0,
                    "endTimeSeconds": 0.75,
                },
                {
                    "clipId": "clip-001",
                    "personInstanceId": "person-1",
                    "ppeItem": "HARD_HAT",
                    "startTimeSeconds": 0.5,
                    "endTimeSeconds": 1.0,
                },
            ],
            "episodes overlap",
        ),
        (
            [
                {
                    "clipId": "clip-001",
                    "personInstanceId": "person-1",
                    "ppeItem": "HARD_HAT",
                    "startTimeSeconds": 2.0,
                    "endTimeSeconds": 3.0,
                }
            ],
            "outside the reviewed frame interval",
        ),
    ],
)
def test_rejects_overlapping_and_out_of_range_episodes(
    tmp_path: Path,
    episodes: list[dict[str, object]],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = _inputs(tmp_path, episodes=episodes)

    assert _run(source, output) == 1
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_rejects_negative_evidence_without_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path, episodes=[])

    assert _run(source, output) == 1
    assert "must belong to exactly one episode" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_media_hash_mismatch_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path, declared_hash="0" * 64)

    assert _run(source, output) == 1
    assert "media SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()


def test_rejects_same_resolved_media_path_under_different_clip_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    duplicate = dict(manifest["clips"][0])
    duplicate["clipId"] = "clip-002"
    manifest["clips"].append(duplicate)
    source.write_text(json.dumps(manifest), encoding="utf-8")

    assert _run(source, output) == 1
    error = capsys.readouterr().err
    assert "clip-001 and clip-002 resolve to the same mediaPath" in error
    assert not output.exists()


def test_rejects_byte_identical_media_at_different_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    copied_media = tmp_path / "copied-clip.mp4"
    copied_media.write_bytes((tmp_path / "clip.mp4").read_bytes())
    duplicate = dict(manifest["clips"][0])
    duplicate["clipId"] = "clip-002"
    duplicate["mediaPath"] = str(copied_media.resolve())
    manifest["clips"].append(duplicate)
    source.write_text(json.dumps(manifest), encoding="utf-8")

    assert _run(source, output) == 1
    error = capsys.readouterr().err
    assert "clip-001 and clip-002 contain identical media bytes" in error
    assert not output.exists()


def test_loader_failure_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, output = _inputs(tmp_path)

    def reject(_path: Path) -> None:
        raise DatasetValidationError("synthetic final-loader rejection")

    monkeypatch.setattr(corpus_tool, "load_evaluation_dataset", reject)

    assert _run(source, output) == 1
    assert "synthetic final-loader rejection" in capsys.readouterr().err
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*"))
