"""OpenCV-backed video source for camera and clip ingestion."""

import asyncio
import importlib

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
        self._capture: object | None = None
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

    async def connect(self) -> None:
        """Load OpenCV and reserve the future connection lifecycle boundary."""
        await asyncio.to_thread(self._load_cv2)
        self._is_connected = False

    async def read_frame(self) -> FrameEnvelope | None:
        """Read one frame from the connected source."""
        raise SourceReadError("OpenCV frame reading is not implemented yet")

    async def close(self) -> None:
        """Release the source if it has been connected."""
        self._capture = None
        self._is_connected = False
