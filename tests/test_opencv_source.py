import asyncio
import importlib
import sys

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.source import SourceConnectionError


def make_config() -> StreamConfig:
    return StreamConfig(
        stream_id="cam-1",
        camera_external_id="ext-cam-1",
        source_url="rtsp://camera.local/live",
    )


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
