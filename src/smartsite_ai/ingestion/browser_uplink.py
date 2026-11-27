"""Decode browser-uplink frames already received from a trusted Backend transport."""

import base64
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
from typing import Literal
from uuid import UUID

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import SourceReadError

_JPEG_PREFIX = "data:image/jpeg;base64,"


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
