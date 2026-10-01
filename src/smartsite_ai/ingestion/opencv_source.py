"""OpenCV-backed video source for camera and clip ingestion."""

import asyncio
import importlib
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError

CAP_PROP_POS_MSEC = 0
CAP_PROP_FPS = 5
DEFAULT_REPLAY_FPS = 30.0


class OpenCvFrameSource:
    """FrameSource implementation backed by ``cv2.VideoCapture``.

    OpenCV is imported lazily so importing this module never initializes camera,
    stream, model, or GPU runtime state.
    """

    def __init__(
        self,
        config: StreamConfig,
        *,
        monotonic_clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._capture: Any | None = None
        self._session_id: UUID | None = None
        self._session_start: datetime | None = None
        self._sequence_number = 0
        self._is_connected = False
        self._io_lock = asyncio.Lock()
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._replay_wall_start: float | None = None
        self._replay_media_start_ms: float | None = None
        self._last_replay_media_ms: float | None = None
        self._last_raw_media_ms: float | None = None
        self._last_effective_media_ms: float | None = None
        self._fallback_frame_interval_ms = 1000.0 / DEFAULT_REPLAY_FPS

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

    @staticmethod
    def _frame_interval_ms(capture: Any) -> float:
        """Return a bounded FPS-derived interval, falling back deterministically."""
        get_fn = getattr(capture, "get", None)
        if get_fn is None:
            return 1000.0 / DEFAULT_REPLAY_FPS
        try:
            fps = float(get_fn(CAP_PROP_FPS))
        except Exception:
            return 1000.0 / DEFAULT_REPLAY_FPS
        if not math.isfinite(fps) or fps < 0.1 or fps > 1000.0:
            return 1000.0 / DEFAULT_REPLAY_FPS
        return 1000.0 / fps

    def _frame_to_envelope(
        self,
        frame: Any,
        *,
        pos_msec: float | None = None,
    ) -> FrameEnvelope:
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

        if not self.config.is_live:
            if self._session_start is None:
                raise SourceReadError("Non-live OpenCV source is missing active session start")
            if pos_msec is None:
                raise SourceReadError("Non-live OpenCV source requires valid media timestamp")
            captured_at = self._session_start + timedelta(milliseconds=pos_msec)
        else:
            captured_at = datetime.now(UTC)

        envelope = FrameEnvelope(
            stream_id=self.config.stream_id,
            session_id=self._session_id,
            camera_external_id=self.config.camera_external_id,
            captured_at=captured_at,
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
        try:
            interval_task = asyncio.create_task(
                asyncio.to_thread(self._frame_interval_ms, capture),
                name=f"opencv-fps-{self.config.stream_id}",
            )
            try:
                fallback_frame_interval_ms = await asyncio.shield(interval_task)
            except asyncio.CancelledError:
                # A to_thread call keeps running after its waiter is cancelled.
                # Wait for the metadata read to finish before releasing capture.
                await asyncio.shield(interval_task)
                raise
            async with self._io_lock:
                self._capture = capture
                self._session_id = uuid4()
                self._session_start = datetime.now(UTC)
                self._sequence_number = 0
                self._replay_wall_start = None
                self._replay_media_start_ms = None
                self._last_replay_media_ms = None
                self._last_raw_media_ms = None
                self._last_effective_media_ms = None
                self._fallback_frame_interval_ms = fallback_frame_interval_ms
                self._is_connected = True
                capture = None
        finally:
            if capture is not None:
                await asyncio.to_thread(self._release_capture, capture)

    async def read_frame(self) -> FrameEnvelope | None:
        """Read one decoded BGR24 frame from the connected source."""
        replay_deadline: float | None = None
        async with self._io_lock:
            if not self._is_connected or self._capture is None:
                raise SourceReadError("OpenCV source is not connected")

            try:
                ok, frame = await asyncio.to_thread(self._capture.read)
            except Exception as exc:
                raise SourceReadError(f"OpenCV failed to read frame: {type(exc).__name__}") from exc

            if not ok or frame is None:
                return None

            pos_msec: float | None = None
            if not self.config.is_live:
                get_fn = getattr(self._capture, "get", None)
                if get_fn is None:
                    raise SourceReadError(
                        "OpenCV capture is missing 'get' method for media timestamps"
                    )
                try:
                    raw_pos = await asyncio.to_thread(get_fn, CAP_PROP_POS_MSEC)
                except Exception as exc:
                    raise SourceReadError(
                        f"OpenCV failed to read media timestamp: {type(exc).__name__}"
                    ) from exc

                if raw_pos is None:
                    raise SourceReadError("OpenCV returned null media timestamp")

                try:
                    val = float(raw_pos)
                except (TypeError, ValueError) as exc:
                    raise SourceReadError(
                        f"OpenCV returned non-numeric media timestamp: {raw_pos!r}"
                    ) from exc

                if not math.isfinite(val) or val < 0.0:
                    raise SourceReadError(
                        f"OpenCV returned invalid media timestamp (finite and >= 0 required): {val}"
                    )
                if self._last_raw_media_ms is not None and val < self._last_raw_media_ms:
                    raise SourceReadError("Replay media timestamp moved backwards")
                if self._last_effective_media_ms is None:
                    pos_msec = val
                elif val == self._last_raw_media_ms:
                    # Some OpenCV/codec combinations repeatedly report 0 ms or
                    # coarse duplicate timestamps. Synthesize a deterministic
                    # frame interval so downstream temporal logic remains strict.
                    pos_msec = self._last_effective_media_ms + self._fallback_frame_interval_ms
                else:
                    pos_msec = max(val, self._last_effective_media_ms + 0.001)
                self._last_raw_media_ms = val
                self._last_effective_media_ms = pos_msec

                if self.config.pace_replay:
                    if self._replay_wall_start is None:
                        self._replay_wall_start = self._monotonic_clock()
                        self._replay_media_start_ms = pos_msec
                        self._last_replay_media_ms = pos_msec
                    if self._replay_media_start_ms is None:
                        raise SourceReadError("Replay pacing is missing its media-time origin")
                    if self._last_replay_media_ms is None:
                        raise SourceReadError(
                            "Replay pacing is missing its previous media timestamp"
                        )
                    media_step_ms = pos_msec - self._last_replay_media_ms
                    if media_step_ms > 10_000:
                        raise SourceReadError("Replay media timestamp gap exceeds 10 seconds")
                    self._last_replay_media_ms = pos_msec
                    media_elapsed = (pos_msec - self._replay_media_start_ms) / 1000.0
                    replay_deadline = self._replay_wall_start + media_elapsed

            envelope = self._frame_to_envelope(frame, pos_msec=pos_msec)

        while replay_deadline is not None:
            delay = replay_deadline - self._monotonic_clock()
            if delay <= 0:
                break
            await self._sleeper(min(delay, 1.0))
        return envelope

    async def close(self) -> None:
        """Release the source if it has been connected."""
        async with self._io_lock:
            capture = self._capture
            self._capture = None
            self._is_connected = False
            self._replay_wall_start = None
            self._replay_media_start_ms = None
            self._last_replay_media_ms = None
            self._last_raw_media_ms = None
            self._last_effective_media_ms = None
            self._fallback_frame_interval_ms = 1000.0 / DEFAULT_REPLAY_FPS
        if capture is not None:
            try:
                await asyncio.to_thread(self._release_capture, capture)
            except Exception as exc:
                raise SourceConnectionError(
                    f"OpenCV failed to release video source: {type(exc).__name__}"
                ) from exc
