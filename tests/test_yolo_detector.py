import asyncio
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from math import inf, nan
from pathlib import Path
from threading import Event, Lock
from uuid import UUID

import pytest

from smartsite_ai.inference.artifacts import VerifiedModelArtifact
from smartsite_ai.inference.yolo import (
    DetectorUnavailableError,
    InferenceResultError,
    RawYoloDetection,
    Yolo11Detector,
    YoloRunnerProtocol,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.runtime.inference_lane import SerializingDetector

SESSION = UUID("00000000-0000-4000-8000-000000000001")
MODEL_SHA256 = "a" * 64
FAKE_MODEL_PATH = Path("/models/yolo11s-ppe.pt").resolve()


def make_frame(**overrides: object) -> FrameEnvelope:
    data: dict[str, object] = {
        "stream_id": "stream-01",
        "session_id": SESSION,
        "camera_external_id": "cam-01",
        "captured_at": datetime(2026, 9, 21, 12, tzinfo=UTC),
        "width": 100,
        "height": 50,
        "sequence_number": 4,
        "payload": bytes(100 * 50 * 3),
        **overrides,
    }
    return FrameEnvelope.model_validate(data)


def make_artifact(**overrides: object) -> VerifiedModelArtifact:
    return VerifiedModelArtifact.model_validate(
        {
            "artifact_id": "yolo11s-ppe",
            "version": "2026.09.21",
            "model_family": "yolo11s",
            "artifact_path": FAKE_MODEL_PATH,
            "sha256": MODEL_SHA256,
            "source_url": "https://models.example.test/yolo11s-ppe.pt",
            "license": "AGPL-3.0",
            "class_map": ((0, "person"), (1, "hard_hat")),
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "image_size": (640, 640),
            "device": "cuda:0",
            "resolved_path": FAKE_MODEL_PATH,
            "actual_sha256": MODEL_SHA256,
            **overrides,
        }
    )


class FakeRunner:
    def __init__(self, rows: Iterable[RawYoloDetection]) -> None:
        self.rows = rows

    def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
        return self.rows


class FailingRunner:
    def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
        raise RuntimeError("provider unavailable")


class BlockingRunner:
    """A native prediction cannot be cancelled by its asyncio waiter."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = Event()
        self._loop = asyncio.get_running_loop()
        self._guard = Lock()
        self.fail_first = fail_first
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
        with self._guard:
            self.calls += 1
            call = self.calls
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if call == 1:
                self._loop.call_soon_threadsafe(self.started.set)
                if not self.release.wait(5):
                    raise RuntimeError("test prediction was not released")
                if self.fail_first:
                    raise RuntimeError("provider failed after cancellation")
            return ()
        finally:
            with self._guard:
                self.in_flight -= 1


@pytest.mark.anyio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_cancelled_prediction_keeps_shared_lane_until_native_work_finishes(
    fail_first: bool,
) -> None:
    runner = BlockingRunner(fail_first=fail_first)
    lane = SerializingDetector(Yolo11Detector(make_artifact(), runner))
    first = asyncio.create_task(lane.detect(make_frame()))
    second = None
    try:
        await asyncio.wait_for(runner.started.wait(), 1)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(lane.detect(make_frame(sequence_number=5)))
        await asyncio.sleep(0.05)
        assert not first.done(), "cancellation returned while native prediction was active"
        assert runner.calls == 1, "another camera entered the active native model"
        runner.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        batch = await asyncio.wait_for(second, 1)
        assert batch.sequence_number == 5
        assert runner.max_in_flight == 1 and runner.in_flight == 0
    finally:
        runner.release.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )


@pytest.mark.anyio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_single_camera_cancellation_returns_only_after_native_prediction_finishes(
    fail_first: bool,
) -> None:
    runner = BlockingRunner(fail_first=fail_first)
    detector = Yolo11Detector(make_artifact(), runner)
    prediction = asyncio.create_task(detector.detect(make_frame()))
    try:
        await asyncio.wait_for(runner.started.wait(), 1)
        prediction.cancel()
        await asyncio.sleep(0)
        prediction.cancel()
        await asyncio.sleep(0.05)
        assert not prediction.done(), "caller could close the runner before prediction finished"
        runner.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(prediction, 1)
        assert runner.in_flight == 0
    finally:
        runner.release.set()
        await asyncio.gather(prediction, return_exceptions=True)


def make_detector(rows: Iterable[RawYoloDetection]) -> Yolo11Detector:
    return Yolo11Detector(make_artifact(), FakeRunner(rows))


def test_runner_protocol_is_structural() -> None:
    runner: YoloRunnerProtocol = FakeRunner(())

    assert isinstance(runner, YoloRunnerProtocol)


def test_detect_preserves_frame_identity_clips_and_normalizes_rows() -> None:
    frame = make_frame()
    detector = make_detector(
        (
            RawYoloDetection(class_id=1, confidence=0.9, x1=-10.0, y1=10.0, x2=110.0, y2=80.0),
            RawYoloDetection(class_id=0, confidence=0.9, x1=20.0, y1=5.0, x2=50.0, y2=20.0),
        )
    )

    batch = asyncio.run(detector.detect(frame))

    assert batch.stream_id == frame.stream_id
    assert batch.session_id == frame.session_id
    assert batch.camera_external_id == frame.camera_external_id
    assert batch.captured_at == frame.captured_at
    assert batch.frame_width == frame.width
    assert batch.frame_height == frame.height
    assert batch.sequence_number == frame.sequence_number
    assert batch.model_artifact_id == "yolo11s-ppe"
    assert batch.model_version == "2026.09.21"
    assert batch.model_sha256 == MODEL_SHA256
    assert [(item.class_id, item.class_name) for item in batch.detections] == [
        (0, "person"),
        (1, "hard_hat"),
    ]
    assert batch.detections[0].bounding_box.model_dump() == {
        "x1": 0.2,
        "y1": 0.1,
        "x2": 0.5,
        "y2": 0.4,
        "coordinate_space": "NORMALIZED_0_1",
    }
    assert batch.detections[1].bounding_box.model_dump() == {
        "x1": 0.0,
        "y1": 0.2,
        "x2": 1.0,
        "y2": 1.0,
        "coordinate_space": "NORMALIZED_0_1",
    }


def test_detect_preserves_empty_results() -> None:
    batch = asyncio.run(make_detector(()).detect(make_frame()))

    assert batch.detections == ()


def test_detect_rejects_unknown_class_ids() -> None:
    detector = make_detector(
        (RawYoloDetection(class_id=9, confidence=0.9, x1=1.0, y1=1.0, x2=2.0, y2=2.0),)
    )

    with pytest.raises(InferenceResultError, match="unknown class ID"):
        asyncio.run(detector.detect(make_frame()))


@pytest.mark.parametrize(
    "row",
    [
        RawYoloDetection(class_id=0, confidence=nan, x1=1.0, y1=1.0, x2=2.0, y2=2.0),
        RawYoloDetection(class_id=0, confidence=0.9, x1=inf, y1=1.0, x2=2.0, y2=2.0),
    ],
)
def test_detect_rejects_non_finite_raw_values(row: RawYoloDetection) -> None:
    with pytest.raises(InferenceResultError, match="finite"):
        asyncio.run(make_detector((row,)).detect(make_frame()))


@pytest.mark.parametrize(
    "row",
    [
        RawYoloDetection(class_id=0, confidence=0.9, x1=2.0, y1=1.0, x2=2.0, y2=2.0),
        RawYoloDetection(class_id=0, confidence=0.9, x1=-10.0, y1=1.0, x2=-1.0, y2=2.0),
    ],
)
def test_detect_rejects_zero_area_and_fully_clipped_boxes(row: RawYoloDetection) -> None:
    with pytest.raises(InferenceResultError, match="positive area"):
        asyncio.run(make_detector((row,)).detect(make_frame()))


def test_detect_rejects_unsupported_pixel_format_before_runner_execution() -> None:
    frame = FrameEnvelope.model_construct(**{**make_frame().model_dump(), "pixel_format": "RGB24"})

    with pytest.raises(DetectorUnavailableError, match="BGR24"):
        asyncio.run(make_detector(()).detect(frame))


def test_detect_wraps_runner_failure() -> None:
    detector = Yolo11Detector(make_artifact(), FailingRunner())

    with pytest.raises(DetectorUnavailableError, match="prediction failed") as exc_info:
        asyncio.run(detector.detect(make_frame()))

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_detect_rejects_1025_raw_rows_without_over_consuming() -> None:
    consumed = 0

    def rows() -> Iterable[RawYoloDetection]:
        nonlocal consumed
        for _ in range(1026):
            consumed += 1
            yield RawYoloDetection(class_id=0, confidence=0.9, x1=1.0, y1=1.0, x2=2.0, y2=2.0)

    with pytest.raises(InferenceResultError, match="1,024"):
        asyncio.run(make_detector(rows()).detect(make_frame()))

    assert consumed == 1025


def test_detect_runs_synchronous_prediction_off_the_event_loop() -> None:
    ticks = 0

    class BlockingRunner:
        def predict(self, frame: FrameEnvelope) -> Iterable[RawYoloDetection]:
            time.sleep(0.05)
            return ()

    async def exercise() -> None:
        nonlocal ticks
        detector = Yolo11Detector(make_artifact(), BlockingRunner())
        prediction = asyncio.create_task(detector.detect(make_frame()))
        while not prediction.done():
            ticks += 1
            await asyncio.sleep(0)
        assert (await prediction).detections == ()

    asyncio.run(exercise())

    assert ticks > 1
