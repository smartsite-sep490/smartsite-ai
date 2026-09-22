import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

import smartsite_ai.tools.detect_video as detect_video
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.tools.detect_video import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_MSEC,
    VideoValidationError,
    run_video_validation,
)


class FakeFrame:
    def __init__(
        self, width: int, height: int, value: int = 0, *, shape: tuple[int, ...] | None = None
    ) -> None:
        self.shape = shape or (height, width, 3)
        self._payload = bytes([value]) * (width * height * 3)

    def tobytes(self) -> bytes:
        return self._payload


class FakeCapture:
    def __init__(
        self,
        frames: Iterable[tuple[float, FakeFrame]],
        *,
        opened: bool = True,
        fps: float = 25.0,
        width: float = 4.0,
        height: float = 2.0,
        read_error: BaseException | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.properties = {
            CAP_PROP_FPS: fps,
            CAP_PROP_FRAME_WIDTH: width,
            CAP_PROP_FRAME_HEIGHT: height,
        }
        self.current_timestamp = 0.0
        self.release_count = 0
        self.read_error = read_error
        self.release_error = release_error

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        if prop == CAP_PROP_POS_MSEC:
            return self.current_timestamp
        return self.properties[prop]

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.read_error is not None:
            raise self.read_error
        if not self.frames:
            return False, None
        self.current_timestamp, frame = self.frames.pop(0)
        return True, frame

    def release(self) -> None:
        self.release_count += 1
        if self.release_error is not None:
            raise self.release_error


class FakeWriter:
    def __init__(
        self,
        *,
        opened: bool = True,
        output_path: Path | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        self.opened = opened
        self.output_path = output_path
        self.frames: list[object] = []
        self.release_count = 0
        self.release_error = release_error
        if output_path is not None:
            output_path.write_bytes(b"incomplete video")

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: object) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.release_count += 1
        if self.release_error is not None:
            raise self.release_error


class FakeDetector:
    def __init__(
        self,
        batches: Iterable[DetectionBatch] | None = None,
        error: BaseException | None = None,
        names_per_frame: Iterable[tuple[str, ...]] | None = None,
    ) -> None:
        self.batches = iter(batches or ())
        self.error = error
        self.names_per_frame = iter(names_per_frame or ())
        self.frames: list[FrameEnvelope] = []

    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        self.frames.append(frame)
        if self.error is not None:
            raise self.error
        if names_per_frame := next(self.names_per_frame, None):
            return make_batch(frame, *names_per_frame)
        return next(self.batches)


class CloseRaisingLoop:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def run_until_complete(self, awaitable: object) -> object:
        return self._loop.run_until_complete(awaitable)

    def close(self) -> None:
        self._loop.close()
        raise RuntimeError("loop close failed")


def make_batch(frame: FrameEnvelope, *names: str) -> DetectionBatch:
    detections = tuple(
        NormalizedDetection(
            class_id=index,
            class_name=name,
            confidence=0.8 + index / 100,
            bounding_box=NormalizedBoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8),
        )
        for index, name in enumerate(names)
    )
    return DetectionBatch.from_frame(
        frame,
        model_artifact_id="demo-model",
        model_version="2026.09.21",
        model_sha256="a" * 64,
        detections=detections,
    )


def make_frame_envelope(sequence: int, captured_at: datetime) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="video-validation",
        session_id=UUID("00000000-0000-4000-8000-000000000001"),
        camera_external_id="local-video",
        captured_at=captured_at,
        width=4,
        height=2,
        sequence_number=sequence,
        payload=bytes(24),
    )


def make_artifact() -> Any:
    from smartsite_ai.inference.artifacts import VerifiedModelArtifact

    return VerifiedModelArtifact.model_validate(
        {
            "artifact_id": "demo-model",
            "version": "2026.09.21",
            "model_family": "yolo11s",
            "artifact_path": Path("C:/models/demo.pt"),
            "sha256": "a" * 64,
            "source_url": "https://models.example.test/demo.pt",
            "license": "AGPL-3.0",
            "class_map": ((0, "person"), (1, "helmet")),
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "image_size": (640, 640),
            "device": "cpu",
            "resolved_path": Path("C:/models/demo.pt"),
            "actual_sha256": "a" * 64,
        }
    )


def run_success(
    tmp_path: Path,
    *,
    frames: Iterable[tuple[float, FakeFrame]] | None = None,
    detector: FakeDetector | None = None,
    capture: FakeCapture | None = None,
    writer: FakeWriter | None = None,
    renderer: Callable[[object, DetectionBatch], object] | None = None,
) -> tuple[
    dict[str, object], FakeCapture, FakeWriter, FakeDetector, list[tuple[object, DetectionBatch]]
]:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    metadata_path = tmp_path / "run.json"
    input_path.write_bytes(b"input")
    capture = capture or FakeCapture(
        frames or ((0.0, FakeFrame(4, 2, 1)), (1_250.0, FakeFrame(4, 2, 2)))
    )
    detector = detector or FakeDetector(names_per_frame=(("person",), ("person", "helmet")))
    render_calls: list[tuple[object, DetectionBatch]] = []
    renderer = renderer or (lambda frame, batch: render_calls.append((frame, batch)) or frame)
    writer = writer or FakeWriter()

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        writer.output_path = path
        return writer

    result = run_video_validation(
        input_path=input_path,
        output_path=output_path,
        metadata_output=metadata_path,
        artifact=make_artifact(),
        class_map={0: "person", 1: "helmet"},
        detector=detector,
        capture_factory=lambda _: capture,
        writer_factory=writer_factory,
        renderer=renderer,
        monotonic_clock=iter((10.0, 12.5)).__next__,
        utc_now_factory=lambda: datetime(2026, 9, 21, 12, tzinfo=UTC),
    )
    return result, capture, writer, detector, render_calls


def test_success_builds_sequential_envelopes_renders_and_replaces_metadata_atomically(
    tmp_path: Path,
) -> None:
    frames = [
        (0.0, FakeFrame(4, 2, 1)),
        (1_250.0, FakeFrame(4, 2, 2)),
    ]
    base = datetime(2026, 9, 21, 12, tzinfo=UTC)
    seed_frames = [
        make_frame_envelope(0, base),
        make_frame_envelope(1, base + timedelta(seconds=1.25)),
    ]
    detector = FakeDetector(names_per_frame=(("person",), ("person", "helmet")))

    result, capture, writer, seen_detector, render_calls = run_success(
        tmp_path, frames=frames, detector=detector
    )

    assert [frame.sequence_number for frame in seen_detector.frames] == [0, 1]
    assert len({frame.session_id for frame in seen_detector.frames}) == 1
    assert all(frame.session_id != seed_frames[0].session_id for frame in seen_detector.frames)
    assert [frame.captured_at for frame in seen_detector.frames] == [
        base,
        base + timedelta(seconds=1.25),
    ]
    assert [frame.payload for frame in seen_detector.frames] == [bytes([1]) * 24, bytes([2]) * 24]
    assert [batch.frame_width for _, batch in render_calls] == [4, 4]
    assert len(render_calls) == 2
    assert writer.frames == [frame for frame, _ in render_calls]
    assert result["frame_count"] == 2
    assert result["detection_count"] == 3
    assert result["detections_by_class"] == {"person": 2, "helmet": 1}
    assert result["processing_fps"] == pytest.approx(0.8)
    assert capture.release_count == 1
    assert writer.release_count == 1
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "input video does not exist"),
        ("directory", "input video must be a regular file"),
        ("extension", "unsupported input video extension"),
        ("input_output", "input and output paths must differ"),
        ("metadata_output", "metadata and output paths must differ"),
        ("existing_output", "output video already exists"),
        ("metadata_extension", "metadata output extension must be .json"),
    ],
)
def test_invalid_paths_are_rejected_before_media_open(
    tmp_path: Path, case: str, message: str
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")
    output_path = tmp_path / "annotated.mp4"
    metadata_path = tmp_path / "run.json"
    if case == "missing":
        input_path.unlink()
    elif case == "directory":
        input_path.unlink()
        input_path.mkdir()
    elif case == "extension":
        input_path.rename(tmp_path / "input.txt")
        input_path = tmp_path / "input.txt"
    elif case == "input_output":
        output_path = input_path
    elif case == "metadata_output":
        metadata_path = output_path
    elif case == "existing_output":
        output_path.write_bytes(b"existing")
    elif case == "metadata_extension":
        metadata_path = tmp_path / "run.txt"

    opened = False

    def capture_factory(_: Path) -> FakeCapture:
        nonlocal opened
        opened = True
        return FakeCapture([])

    with pytest.raises(VideoValidationError, match=message):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=metadata_path,
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(),
            capture_factory=capture_factory,
            writer_factory=lambda *_: FakeWriter(),
            renderer=lambda frame, batch: frame,
        )
    assert opened is False


def test_hard_linked_output_is_rejected_without_opening_media(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "linked.mp4"
    input_path.write_bytes(b"input")
    try:
        os.link(input_path, output_path)
    except OSError as error:
        pytest.skip(f"hard links are unavailable on this platform: {error}")

    opened = False

    def capture_factory(_: Path) -> FakeCapture:
        nonlocal opened
        opened = True
        return FakeCapture([])

    with pytest.raises(VideoValidationError, match="input and output paths must differ"):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(),
            capture_factory=capture_factory,
            writer_factory=lambda *_: FakeWriter(),
            renderer=lambda frame, batch: frame,
        )
    assert opened is False


@pytest.mark.parametrize(
    "failure", ["capture", "fps", "dimensions", "writer", "shape", "inference", "empty"]
)
def test_all_media_and_processing_failures_release_resources_and_leave_no_metadata(
    tmp_path: Path, failure: str
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    metadata_path = tmp_path / "run.json"
    input_path.write_bytes(b"input")
    capture = FakeCapture(
        [] if failure in {"capture", "empty"} else [(0.0, FakeFrame(4, 2))],
        opened=failure != "capture",
        fps=0.0 if failure == "fps" else 25.0,
        width=0.0 if failure == "dimensions" else 4.0,
    )
    writer = FakeWriter(opened=failure != "writer")
    detector = FakeDetector(error=RuntimeError("inference failed"))
    if failure == "shape":
        capture.frames = [(0.0, FakeFrame(4, 2, shape=(2, 4)))]
    if failure == "empty":
        capture.frames = []

    def capture_factory(_: Path) -> FakeCapture:
        return capture

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        writer.output_path = path
        return writer

    with pytest.raises(VideoValidationError):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=metadata_path,
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=detector,
            capture_factory=capture_factory,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
            monotonic_clock=lambda: 1.0,
            utc_now_factory=lambda: datetime(2026, 9, 21, 12, tzinfo=UTC),
        )

    assert capture.release_count == 1
    if failure in {"writer", "shape", "inference", "empty"}:
        assert writer.release_count == 1
    else:
        assert writer.release_count == 0
    assert not metadata_path.exists()
    assert not output_path.exists()


def test_existing_metadata_is_preserved_when_run_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text('{"status":"previous"}', encoding="utf-8")
    capture = FakeCapture([(0.0, FakeFrame(4, 2))])

    with pytest.raises(VideoValidationError):
        run_video_validation(
            input_path=input_path,
            output_path=(tmp_path / "annotated.mp4"),
            metadata_output=metadata_path,
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(error=RuntimeError("boom")),
            capture_factory=lambda _: capture,
            writer_factory=lambda path, codec, fps, size: FakeWriter(output_path=path),
            renderer=lambda frame, batch: frame,
        )

    assert metadata_path.read_text(encoding="utf-8") == '{"status":"previous"}'


@pytest.mark.parametrize("cleanup_error", ["writer", "capture"])
def test_cleanup_release_errors_do_not_skip_other_release_or_output_cleanup(
    tmp_path: Path, cleanup_error: str
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    input_path.write_bytes(b"input")
    capture = FakeCapture(
        [(0.0, FakeFrame(4, 2))],
        release_error=RuntimeError("capture release failed")
        if cleanup_error == "capture"
        else None,
    )
    writer = FakeWriter(
        release_error=RuntimeError("writer release failed") if cleanup_error == "writer" else None
    )

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        return writer

    with pytest.raises(VideoValidationError):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(error=RuntimeError("inference failed")),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
        )

    assert capture.release_count == 1
    assert writer.release_count == 1
    assert not output_path.exists()


def test_event_loop_close_error_does_not_skip_media_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    input_path.write_bytes(b"input")
    capture = FakeCapture([(0.0, FakeFrame(4, 2))])
    writer = FakeWriter()
    real_new_event_loop = asyncio.new_event_loop

    def new_raising_loop() -> CloseRaisingLoop:
        return CloseRaisingLoop(real_new_event_loop())

    monkeypatch.setattr(detect_video.asyncio, "new_event_loop", new_raising_loop)

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        return writer

    with pytest.raises(VideoValidationError):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(error=RuntimeError("inference failed")),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
        )

    assert capture.release_count == 1
    assert writer.release_count == 1
    assert not output_path.exists()


def test_keyboard_interrupt_still_releases_media_and_removes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    input_path.write_bytes(b"input")
    capture = FakeCapture([(0.0, FakeFrame(4, 2))], read_error=KeyboardInterrupt())
    writer = FakeWriter()

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        return writer

    with pytest.raises(KeyboardInterrupt):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=FakeDetector(),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
        )

    assert capture.release_count == 1
    assert writer.release_count == 1
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("session_id", UUID("00000000-0000-4000-8000-000000000099"), "identity"),
        ("sequence_number", 99, "identity"),
        ("model_artifact_id", "other-model", "artifact"),
        ("model_sha256", "b" * 64, "artifact"),
    ],
)
def test_detection_batch_identity_and_verified_artifact_are_checked(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    input_path.write_bytes(b"input")
    capture = FakeCapture([(0.0, FakeFrame(4, 2))])

    class MismatchDetector(FakeDetector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            self.frames.append(frame)
            return make_batch(frame, "person").model_copy(update={field: value})

    writer = FakeWriter()

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        return writer

    with pytest.raises(VideoValidationError, match=message):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=MismatchDetector(),
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
        )

    assert capture.release_count == 1
    assert writer.release_count == 1
    assert not output_path.exists()


def _required_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--input",
        str(tmp_path / "input.mp4"),
        "--model",
        str(tmp_path / "model.pt"),
        "--model-artifact-id",
        "demo-model",
        "--model-version",
        "2026.09.21",
        "--model-family",
        "yolo11s",
        "--model-sha256",
        "a" * 64,
        "--class-map",
        str(tmp_path / "classes.json"),
        "--output",
        str(tmp_path / "annotated.mp4"),
        "--metadata-output",
        str(tmp_path / "run.json"),
    ]


def test_parser_requires_explicit_model_provenance_and_license(tmp_path: Path) -> None:
    args = _required_cli_args(tmp_path)
    for incomplete_args in (
        args + ["--model-license", "AGPL-3.0"],
        args + ["--model-source-url", "https://models.example.test/demo.pt"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            detect_video.build_parser().parse_args(incomplete_args)
        assert exc_info.value.code == 2

    parsed = detect_video.build_parser().parse_args(
        args
        + [
            "--model-source-url",
            "https://models.example.test/demo.pt",
            "--model-license",
            "AGPL-3.0",
        ]
    )
    assert parsed.model_source_url == "https://models.example.test/demo.pt"
    assert parsed.model_license == "AGPL-3.0"


def test_cli_rejects_placeholder_provenance_before_model_or_video_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _required_cli_args(tmp_path) + [
        "--model-source-url",
        "https://local.invalid/model",
        "--model-license",
        "UNSPECIFIED",
    ]

    assert detect_video.run(args) == 1
    assert "provenance" in capsys.readouterr().err


def test_cli_rejects_a_symlink_model_before_provider_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"weight")
    link = tmp_path / "model-link.pt"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable on this platform: {error}")
    (tmp_path / "classes.json").write_text('{"0": "person"}\n', encoding="utf-8")
    constructed: list[str] = []

    class RecordingRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append("init")

        def load(self, _artifact: object) -> None:
            constructed.append("load")

        def close(self) -> None:
            return None

    monkeypatch.setattr(detect_video, "UltralyticsYoloRunner", RecordingRunner)
    monkeypatch.chdir(tmp_path)
    args = [
        "--input",
        "input.mp4",
        "--model",
        "model-link.pt",
        "--model-artifact-id",
        "demo-model",
        "--model-version",
        "2026.09.21",
        "--model-family",
        "yolo11s",
        "--model-sha256",
        "a" * 64,
        "--class-map",
        "classes.json",
        "--output",
        "annotated.mp4",
        "--metadata-output",
        "run.json",
        "--model-source-url",
        "https://models.example.test/demo.pt",
        "--model-license",
        "AGPL-3.0",
    ]

    assert detect_video.run(args) == 1
    assert constructed == []
    assert "symlink" in capsys.readouterr().err


@pytest.mark.parametrize(
    "payload",
    [
        '{"0": "person", "0": "helmet"}\n',
        '[[true, "helmet"]]\n',
        '[[0.9, "person"]]\n',
        '[[1.0, "person"]]\n',
    ],
)
def test_cli_rejects_ambiguous_class_map_values(
    tmp_path: Path, payload: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "classes.json").write_text(payload, encoding="utf-8")
    args = _required_cli_args(tmp_path) + [
        "--model-source-url",
        "https://models.example.test/demo.pt",
        "--model-license",
        "AGPL-3.0",
    ]

    assert detect_video.run(args) == 1
    assert "class map" in capsys.readouterr().err


def test_class_map_parser_accepts_only_canonical_integer_ids(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    path.write_text('{"0": "person", "1": "helmet"}\n', encoding="utf-8")

    assert detect_video._parse_class_map(path) == {0: "person", 1: "helmet"}

    path.write_text('[[0, "person"], ["1", "helmet"]]\n', encoding="utf-8")

    assert detect_video._parse_class_map(path) == {0: "person", 1: "helmet"}


def test_cli_does_not_replace_the_class_map_with_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class_map = tmp_path / "classes.json"
    original = b'{"0": "person"}\n'
    class_map.write_bytes(original)
    constructed: list[str] = []

    class RecordingRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append("init")

        def load(self, _artifact: object) -> None:
            constructed.append("load")

        def close(self) -> None:
            return None

    monkeypatch.setattr(detect_video, "UltralyticsYoloRunner", RecordingRunner)
    args = [
        item if item != str(tmp_path / "run.json") else str(class_map)
        for item in _required_cli_args(tmp_path)
    ]
    args += [
        "--model-source-url",
        "https://models.example.test/demo.pt",
        "--model-license",
        "AGPL-3.0",
    ]

    assert detect_video.run(args) == 1
    assert class_map.read_bytes() == original
    assert constructed == []
    assert "configuration" in capsys.readouterr().err


def test_keyboard_interrupt_during_inference_joins_the_worker_before_returning(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    input_path.write_bytes(b"input")
    capture = FakeCapture([(0.0, FakeFrame(4, 2))])
    writer = FakeWriter()

    class BlockingDetector:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.finished = threading.Event()

        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            asyncio.ensure_future(asyncio.to_thread(self._block))
            while not self.started.is_set():
                await asyncio.sleep(0.01)
            raise KeyboardInterrupt

        def _block(self) -> None:
            self.started.set()
            time.sleep(0.2)
            self.finished.set()

    detector = BlockingDetector()

    def writer_factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        path.write_bytes(b"incomplete video")
        return writer

    with pytest.raises(KeyboardInterrupt):
        run_video_validation(
            input_path=input_path,
            output_path=output_path,
            metadata_output=tmp_path / "run.json",
            artifact=make_artifact(),
            class_map={0: "person", 1: "helmet"},
            detector=detector,
            capture_factory=lambda _: capture,
            writer_factory=writer_factory,
            renderer=lambda frame, batch: frame,
        )

    assert detector.finished.is_set()
    assert capture.release_count == 1
    assert writer.release_count == 1
    assert not output_path.exists()
