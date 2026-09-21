"""OpenCV-backed video source for camera and clip ingestion."""

import asyncio
import importlib
from datetime import UTC, datetime
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

    def _frame_to_envelope(self, frame: Any) -> FrameEnvelope:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        if len(shape or ()) != 3:
            raise SourceReadError("OpenCV frame must be a height x width x 3 BGR image")

        height, width, channels = shape
        if channels != 3 or width <= 0 or height <= 0:
            raise SourceReadError("OpenCV frame must be a non-empty 3-channel BGR image")

        if str(dtype) != "uint8":
            raise SourceReadError("OpenCV frame must use uint8 BGR24 pixels")

        flags = getattr(frame, "flags", None)
        is_contiguous = getattr(flags, "c_contiguous", False)
        if not is_contiguous and hasattr(flags, "get"):
            is_contiguous = flags.get("C_CONTIGUOUS", False)

        if not is_contiguous:
            frame = frame.copy(order="C")

        payload = frame.tobytes()
        if self._session_id is None:
            raise SourceReadError("OpenCV source is missing an active session")

        envelope = FrameEnvelope(
            stream_id=self.config.stream_id,
            session_id=self._session_id,
            camera_external_id=self.config.camera_external_id,
            captured_at=datetime.now(UTC),
            width=int(width),
            height=int(height),
            sequence_number=self._sequence_number,
            payload=payload,
        )
        self._sequence_number += 1
        return envelope

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
        """Read one decoded BGR24 frame from the connected source."""
        if not self._is_connected or self._capture is None:
            raise SourceReadError("OpenCV source is not connected")

        try:
            ok, frame = await asyncio.to_thread(self._capture.read)
        except Exception as exc:
            raise SourceReadError(f"OpenCV failed to read frame: {type(exc).__name__}") from exc

        if not ok or frame is None:
            return None

        return self._frame_to_envelope(frame)

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
