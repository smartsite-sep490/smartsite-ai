"""Decode browser-uplink frames already received from a trusted Backend transport."""

import asyncio
import base64
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from datetime import datetime
from io import BytesIO
from typing import Literal
from uuid import UUID

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import (
    SourceConnectionError,
    SourceReadError,
    SourceTimeoutError,
)

_JPEG_PREFIX = "data:image/jpeg;base64,"


class BrowserUplinkFrameSource:
    """Read decoded frames from a Backend packet stream scoped to one assigned lease."""

    def __init__(
        self,
        packets: Callable[[], AsyncGenerator[Mapping[str, object], None]],
        *,
        stream_id: str,
        camera_external_id: str,
        session_id: UUID,
    ) -> None:
        self._packets = packets
        self._stream_id = stream_id
        self._camera_external_id = camera_external_id
        self._session_id = session_id
        self._stream: AsyncGenerator[Mapping[str, object], None] | None = None
        self._pending: asyncio.Task[FrameEnvelope | None] | None = None
        self._last_sequence = 0

    @classmethod
    def from_frame_reader(
        cls,
        read_packet: Callable[[int], Awaitable[Mapping[str, object] | None]],
        *,
        stream_id: str,
        camera_external_id: str,
        session_id: UUID,
    ) -> "BrowserUplinkFrameSource":
        async def packets() -> AsyncGenerator[Mapping[str, object], None]:
            while True:
                try:
                    packet = await read_packet(source._last_sequence)
                except ValueError:
                    raise SourceReadError("Browser frame transport failed") from None
                if packet is not None:
                    yield packet
                else:
                    await asyncio.sleep(0.15)

        source = cls(
            packets,
            stream_id=stream_id,
            camera_external_id=camera_external_id,
            session_id=session_id,
        )
        return source

    @property
    def source_id(self) -> str:
        return self._stream_id

    @property
    def assigned_session_id(self) -> UUID:
        return self._session_id

    @property
    def is_connected(self) -> bool:
        return self._stream is not None

    async def connect(self) -> None:
        if self._stream is not None or self._pending is not None:
            raise SourceConnectionError("browser uplink source is already connected")
        self._stream = self._packets()

    async def read_frame(self) -> FrameEnvelope | None:
        stream = self._stream
        if stream is None or self._pending is not None:
            raise SourceReadError("browser uplink source cannot start a read")
        pending = asyncio.create_task(self._next_frame(stream))
        self._pending = pending
        try:
            frame = await pending
            if self._stream is not stream or frame is None:
                return None
            if frame.sequence_number <= self._last_sequence:
                raise SourceReadError("Browser frame sequence did not advance")
            self._last_sequence = frame.sequence_number
            return frame
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if self._stream is not stream and caller is not None and not caller.cancelling():
                return None
            raise
        finally:
            self._pending = None

    async def _next_frame(
        self, stream: AsyncGenerator[Mapping[str, object], None]
    ) -> FrameEnvelope | None:
        try:
            async with asyncio.timeout(2):
                packet = await anext(stream)
        except StopAsyncIteration:
            return None
        except TimeoutError:
            raise SourceTimeoutError("browser uplink did not supply a frame in time") from None
        return await asyncio.to_thread(
            decode_browser_uplink_frame,
            packet,
            stream_id=self._stream_id,
            camera_external_id=self._camera_external_id,
            session_id=self._session_id,
        )

    async def close(self) -> None:
        stream = self._stream
        self._stream = None
        if self._pending is not None:
            self._pending.cancel()
            await asyncio.gather(self._pending, return_exceptions=True)
        if stream is not None:
            await stream.aclose()


class _MediaPacket(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["media-frame"]
    media_version: int = Field(ge=1, le=1, alias="mediaVersion")
    camera_external_id: str = Field(min_length=1, max_length=128, alias="cameraExternalId")
    session_id: str = Field(
        pattern=r"^[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$",
        alias="sessionId",
    )
    sequence_number: str = Field(pattern=r"^[1-9][0-9]{0,18}$", alias="sequenceNumber")
    captured_at: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$",
        alias="capturedAt",
    )
    width: int = Field(ge=1, le=640)
    height: int = Field(ge=1, le=640)
    image_data_url: str = Field(max_length=180_000, alias="imageDataUrl")


def decode_browser_uplink_frame(
    value: Mapping[str, object],
    *,
    stream_id: str,
    camera_external_id: str,
    session_id: UUID,
) -> FrameEnvelope:
    """Validate the scoped JPEG envelope and preserve unmodified capture lineage."""
    try:
        packet = _MediaPacket.model_validate(value)
        if (
            packet.camera_external_id != camera_external_id
            or UUID(packet.session_id) != session_id
            or not packet.image_data_url.startswith(_JPEG_PREFIX)
        ):
            raise ValueError("wrong source or encoding")
        sequence_number = int(packet.sequence_number)
        if sequence_number > 9_223_372_036_854_775_807:
            raise ValueError("sequence out of range")
        captured_at = datetime.fromisoformat(packet.captured_at)
        jpeg = base64.b64decode(packet.image_data_url[len(_JPEG_PREFIX) :], validate=True)
        with Image.open(BytesIO(jpeg), formats=("JPEG",)) as image:
            if image.size != (packet.width, packet.height):
                raise ValueError("declared dimensions do not match JPEG")
            payload = image.convert("RGB").tobytes("raw", "BGR")
        return FrameEnvelope(
            stream_id=stream_id,
            session_id=session_id,
            camera_external_id=camera_external_id,
            captured_at=captured_at,
            width=packet.width,
            height=packet.height,
            sequence_number=sequence_number,
            payload=payload,
        )
    except (ValueError, OSError, Image.DecompressionBombError):
        raise SourceReadError("invalid browser uplink frame") from None
