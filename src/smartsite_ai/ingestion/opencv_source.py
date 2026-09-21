"""OpenCV-backed video source for camera and clip ingestion."""

import asyncio
import importlib
from typing import Any
from uuid import UUID, uuid4

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError


class OpenCvFrameSource:
    """FrameSource implementation backed by ``cv2.VideoCapture``.

    OpenCV is imported lazily so importing this module never initializes camera,
    stream, model, or GPU runtime state.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._capture: Any | None = None
        self._session_id: UUID | None = None
        self._sequence_number = 0
        self._is_connected = False

    @property
    def source_id(self) -> str:
        return self.config.stream_id

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def _load_cv2() -> object:
        try:
            return importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise SourceConnectionError(
                "OpenCV runtime is unavailable for video source ingestion."
            ) from exc

    def _resolve_capture_source(self) -> int | str:
        raw_source = self.config.get_raw_source_url().strip()
        if raw_source.isdecimal():
            return int(raw_source)
        return raw_source

    def _open_capture(self, cv2: Any) -> Any:
        source = self._resolve_capture_source()
        try:
            capture = cv2.VideoCapture(source)
        except Exception as exc:
            raise SourceConnectionError(
                f"OpenCV failed to create VideoCapture for {self.config.sanitized_source_url}: "
                f"{type(exc).__name__}"
            ) from exc

        try:
            is_opened = bool(capture.isOpened())
        except Exception as exc:
            self._release_capture(capture)
            raise SourceConnectionError(
                f"OpenCV failed to validate VideoCapture for {self.config.sanitized_source_url}: "
                f"{type(exc).__name__}"
            ) from exc

        if not is_opened:
            self._release_capture(capture)
            raise SourceConnectionError(
                f"OpenCV could not open video source {self.config.sanitized_source_url}"
            )

        return capture

    @staticmethod
    def _release_capture(capture: Any) -> None:
        release = getattr(capture, "release", None)
        if release is not None:
            release()

    async def connect(self) -> None:
        """Open the configured video source and start a fresh stream session."""
        await self.close()
        cv2 = await asyncio.to_thread(self._load_cv2)
        capture = await asyncio.to_thread(self._open_capture, cv2)

        self._capture = capture
        self._session_id = uuid4()
        self._sequence_number = 0
        self._is_connected = True

    async def read_frame(self) -> FrameEnvelope | None:
        """Read one frame from the connected source."""
        raise SourceReadError("OpenCV frame reading is not implemented yet")

    async def close(self) -> None:
        """Release the source if it has been connected."""
        capture = self._capture
        self._capture = None
        self._is_connected = False
        if capture is not None:
            try:
                await asyncio.to_thread(self._release_capture, capture)
            except Exception as exc:
                raise SourceConnectionError(
                    f"OpenCV failed to release video source: {type(exc).__name__}"
                ) from exc
