import asyncio
import importlib
import sys
from collections.abc import Callable

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import SourceConnectionError


class FakeCapture:
    def __init__(self, *, opened: bool = True, release_raises: bool = False) -> None:
        self.opened = opened
        self.release_raises = release_raises
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self.opened

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
