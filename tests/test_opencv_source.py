import asyncio
import importlib
import sys
from collections.abc import Callable, Iterable

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError


class FakeFrame:
    def __init__(
        self,
        data: bytes,
        *,
        shape: tuple[int, ...],
        dtype: str = "uint8",
        contiguous: bool = True,
        copied_data: bytes | None = None,
    ) -> None:
        self._data = data
        self._copied_data = copied_data or data
        self.shape = shape
        self.dtype = dtype
        self.flags = {"C_CONTIGUOUS": contiguous}
        self.copy_calls = 0

    def copy(self, *, order: str) -> "FakeFrame":
        self.copy_calls += 1
        assert order == "C"
        return FakeFrame(
            self._copied_data,
            shape=self.shape,
            dtype=self.dtype,
            contiguous=True,
        )

    def tobytes(self) -> bytes:
        return self._data


class FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        release_raises: bool = False,
        frames: Iterable[object] = (),
    ) -> None:
        self.opened = opened
        self.release_raises = release_raises
        self.release_calls = 0
        self.read_calls = 0
        self.frames = list(frames)

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        self.read_calls += 1
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return True, frame

    def release(self) -> None:
        self.release_calls += 1
        if self.release_raises:
            raise RuntimeError("simulated release failure")
        self.opened = False


class FakeCv2:
    def __init__(self, capture_factory: Callable[[int | str], FakeCapture]) -> None:
        self.capture_factory = capture_factory
        self.video_capture_inputs: list[int | str] = []

    def VideoCapture(self, source: int | str) -> FakeCapture:
        self.video_capture_inputs.append(source)
        return self.capture_factory(source)


def make_config(source_url: str = "rtsp://camera.local/live") -> StreamConfig:
    return StreamConfig(
        stream_id="cam-1",
        camera_external_id="ext-cam-1",
        source_url=source_url,
    )


def make_bgr_frame(width: int, height: int, fill: int = 0) -> FakeFrame:
    return FakeFrame(bytes([fill]) * width * height * 3, shape=(height, width, 3))


def install_fake_cv2(monkeypatch: pytest.MonkeyPatch, cv2: FakeCv2) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "cv2":
            return cv2
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def test_opencv_source_module_import_does_not_load_cv2() -> None:
    sys.modules.pop("cv2", None)

    import smartsite_ai.ingestion.opencv_source  # noqa: F401

    assert "cv2" not in sys.modules


def test_opencv_frame_source_exposes_protocol_shape() -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    source = OpenCvFrameSource(make_config())

    assert source.source_id == "cam-1"
    assert source.is_connected is False
    assert hasattr(source, "connect")
    assert hasattr(source, "read_frame")
    assert hasattr(source, "close")


def test_connect_without_cv2_raises_clear_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    source = OpenCvFrameSource(make_config())

    with pytest.raises(SourceConnectionError, match="OpenCV runtime is unavailable"):
        asyncio.run(source.connect())


def test_connect_opens_capture_and_starts_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    capture = FakeCapture(opened=True)
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())

    assert cv2.video_capture_inputs == ["rtsp://camera.local/live"]
    assert source.is_connected is True
    assert source._capture is capture
    assert source._session_id is not None
    assert source._sequence_number == 0


def test_connect_converts_numeric_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    cv2 = FakeCv2(lambda _source: FakeCapture(opened=True))
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config("0"))

    asyncio.run(source.connect())

    assert cv2.video_capture_inputs == [0]


def test_connect_keeps_video_path_and_url_as_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    cv2 = FakeCv2(lambda _source: FakeCapture(opened=True))
    install_fake_cv2(monkeypatch, cv2)

    asyncio.run(OpenCvFrameSource(make_config(r"D:\data\clip.mp4")).connect())
    asyncio.run(OpenCvFrameSource(make_config("rtsp://camera.local/live")).connect())

    assert cv2.video_capture_inputs == [r"D:\data\clip.mp4", "rtsp://camera.local/live"]


def test_connect_failure_releases_capture_and_masks_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    capture = FakeCapture(opened=False)
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(
        make_config("rtsp://admin:super-secret@camera.local/live?token=secret-token")
    )

    with pytest.raises(SourceConnectionError) as exc_info:
        asyncio.run(source.connect())

    message = str(exc_info.value)
    assert "super-secret" not in message
    assert "secret-token" not in message
    assert "***" in message
    assert capture.release_calls == 1
    assert source.is_connected is False


def test_close_releases_capture_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    capture = FakeCapture(opened=True)
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())
    asyncio.run(source.close())
    asyncio.run(source.close())

    assert source.is_connected is False
    assert source._capture is None
    assert capture.release_calls == 1


def test_read_frame_returns_bgr24_frame_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    frame = FakeFrame(bytes(range(18)), shape=(2, 3, 3))
    capture = FakeCapture(opened=True, frames=[frame])
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())
    envelope = asyncio.run(source.read_frame())

    assert envelope is not None
    assert envelope.stream_id == "cam-1"
    assert envelope.camera_external_id == "ext-cam-1"
    assert envelope.session_id == source._session_id
    assert envelope.width == 3
    assert envelope.height == 2
    assert envelope.sequence_number == 0
    assert envelope.pixel_format == "BGR24"
    assert envelope.payload == bytes(range(18))
    assert len(envelope.payload) == envelope.width * envelope.height * 3
    assert source._sequence_number == 1


def test_read_frame_sequence_increments_and_eof_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    frames = [make_bgr_frame(1, 1, fill=value) for value in (1, 2, 3)]
    capture = FakeCapture(opened=True, frames=frames)
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())
    envelopes = [asyncio.run(source.read_frame()) for _ in range(3)]
    eof = asyncio.run(source.read_frame())

    assert [envelope.sequence_number for envelope in envelopes if envelope is not None] == [0, 1, 2]
    assert eof is None
    assert source._sequence_number == 3


def test_read_frame_before_connect_raises_read_error() -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    source = OpenCvFrameSource(make_config())

    with pytest.raises(SourceReadError, match="not connected"):
        asyncio.run(source.read_frame())


@pytest.mark.parametrize(
    "bad_frame",
    [
        FakeFrame(bytes(4), shape=(2, 2)),
        FakeFrame(bytes(16), shape=(2, 2, 4)),
        FakeFrame(bytes(12), shape=(2, 2, 3), dtype="float32"),
    ],
)
def test_read_frame_rejects_invalid_bgr24_frame(
    monkeypatch: pytest.MonkeyPatch,
    bad_frame: FakeFrame,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    capture = FakeCapture(opened=True, frames=[bad_frame])
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())

    with pytest.raises(SourceReadError):
        asyncio.run(source.read_frame())
    assert source._sequence_number == 0


def test_read_frame_copies_non_contiguous_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

    packed = bytes(range(18))
    frame = FakeFrame(
        b"non-contiguous-view",
        shape=(2, 3, 3),
        contiguous=False,
        copied_data=packed,
    )
    capture = FakeCapture(opened=True, frames=[frame])
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    source = OpenCvFrameSource(make_config())

    asyncio.run(source.connect())
    envelope = asyncio.run(source.read_frame())

    assert envelope is not None
    assert frame.copy_calls == 1
    assert envelope.width == 3
    assert envelope.height == 2
    assert envelope.payload == packed


async def wait_until(predicate: Callable[[], bool], max_iterations: int = 2000) -> None:
    for _ in range(max_iterations):
        if predicate():
            return
        await asyncio.sleep(0)
    raise TimeoutError("Predicate condition not met within bounded cooperative iterations")


@pytest.mark.anyio
async def test_stream_worker_stops_cleanly_after_finite_opencv_clip_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
    from smartsite_ai.ingestion.queue import QueueClosedError
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.ingestion.worker import StreamWorker

    capture = FakeCapture(
        opened=True,
        frames=[make_bgr_frame(2, 2, fill=1), make_bgr_frame(2, 2, fill=2)],
    )
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    config = StreamConfig(
        stream_id="clip-1",
        camera_external_id="ext-clip-1",
        source_url=r"D:\data\clip.mp4",
        is_live=False,
    )
    source = OpenCvFrameSource(config)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    frame1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    frame2 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    await wait_until(lambda: capture.release_calls == 1)

    assert worker.state == StreamState.STOPPED
    assert frame1.sequence_number == 0
    assert frame2.sequence_number == 1
    assert frame1.session_id == frame2.session_id
    assert frame1.payload == bytes([1]) * 12
    assert frame2.payload == bytes([2]) * 12
    assert worker.snapshot().metrics.frames_enqueued == 2
    assert worker.snapshot().metrics.reconnect_attempts == 0
    with pytest.raises(QueueClosedError):
        await worker.get_frame()


@pytest.mark.anyio
async def test_stream_worker_reconnects_live_opencv_source_after_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.ingestion.testing import FakeSleeper
    from smartsite_ai.ingestion.worker import StreamWorker

    captures = [
        FakeCapture(opened=True, frames=[make_bgr_frame(1, 1, fill=1)]),
        FakeCapture(opened=True, frames=[make_bgr_frame(1, 1, fill=2)]),
    ]

    def capture_factory(_source: int | str) -> FakeCapture:
        return captures.pop(0)

    cv2 = FakeCv2(capture_factory)
    install_fake_cv2(monkeypatch, cv2)
    sleeper = FakeSleeper()
    config = StreamConfig(
        stream_id="live-1",
        camera_external_id="ext-live-1",
        source_url="rtsp://camera.local/live",
        reconnect_initial_delay=1.0,
        reconnect_backoff_factor=2.0,
        reconnect_jitter=0.0,
        max_consecutive_failures=2,
    )
    source = OpenCvFrameSource(config)
    worker = StreamWorker(config=config, source=source, sleeper=sleeper)

    await worker.start()
    try:
        first = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        second = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        await wait_until(lambda: worker.state == StreamState.ERROR)

        assert first.sequence_number == 0
        assert second.sequence_number == 0
        assert first.session_id != second.session_id
        assert worker.snapshot().metrics.connection_errors == 2
        assert worker.snapshot().metrics.reconnect_attempts == 2
        assert worker.snapshot().metrics.last_error == "connection_closed_eof"
        assert sleeper.sleep_calls == [1.0]
        assert cv2.video_capture_inputs == [
            "rtsp://camera.local/live",
            "rtsp://camera.local/live",
        ]
    finally:
        await worker.stop()


@pytest.mark.anyio
async def test_stream_worker_applies_sampling_to_opencv_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
    from smartsite_ai.ingestion.status import StreamState
    from smartsite_ai.ingestion.worker import StreamWorker

    capture = FakeCapture(
        opened=True,
        frames=[
            make_bgr_frame(1, 1, fill=1),
            make_bgr_frame(1, 1, fill=2),
            make_bgr_frame(1, 1, fill=3),
        ],
    )
    cv2 = FakeCv2(lambda _source: capture)
    install_fake_cv2(monkeypatch, cv2)
    config = StreamConfig(
        stream_id="sampled-clip",
        camera_external_id="ext-sampled-clip",
        source_url=r"D:\data\clip.mp4",
        target_fps=1.0,
        is_live=False,
    )
    source = OpenCvFrameSource(config)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    received = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    await wait_until(lambda: worker.state == StreamState.STOPPED)

    status = worker.snapshot()
    assert received.sequence_number == 0
    assert status.metrics.frames_enqueued == 1
    assert status.metrics.sampled_out_frames == 2
    assert capture.read_calls == 4
