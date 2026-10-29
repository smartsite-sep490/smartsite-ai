"""Purpose-specific configuration JPEG encoding; no detector or capture imports."""

import base64
import hashlib
import io
from dataclasses import dataclass
from typing import Literal, TypedDict

from smartsite_ai.ingestion.envelope import FrameEnvelope

SourceKind = Literal["LIVE", "REPLAY"]
MAX_JPEG_BYTES = 1024 * 1024
MAX_INPUT_PIXELS = 16_777_216


class ConfigurationFrame(TypedDict):
    cameraExternalId: str
    sessionId: str
    sequenceNumber: str
    capturedAt: str
    width: int
    height: int
    imageDataUrl: str
    imageSha256: str


class ConfigurationImagePayload(TypedDict):
    schemaVersion: Literal[1]
    purpose: Literal["REGION_CONFIGURATION"]
    sourceKind: SourceKind
    frame: ConfigurationFrame


@dataclass(frozen=True)
class ConfigurationJpeg:
    data: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.data, bytes)
            or not 0 < len(self.data) <= MAX_JPEG_BYTES
            or not self.data.startswith(b"\xff\xd8")
            or not self.data.endswith(b"\xff\xd9")
            or type(self.width) is not int
            or type(self.height) is not int
            or not 64 <= self.width <= 1280
            or not 64 <= self.height <= 1280
        ):
            raise ValueError("Configuration image exceeds its JPEG or dimension budget")


def encode_configuration_image(frame: FrameEnvelope) -> ConfigurationJpeg:
    """Run in a worker thread. Preserve aspect ratio, never upscale small inputs."""
    if min(frame.width, frame.height) < 64 or frame.width * frame.height > MAX_INPUT_PIXELS:
        raise ValueError("Configuration image exceeds its input budget")
    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
    image.thumbnail((1280, 1280))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return ConfigurationJpeg(output.getvalue(), image.width, image.height)


def configuration_payload(
    frame: FrameEnvelope, kind: SourceKind, jpeg: ConfigurationJpeg
) -> ConfigurationImagePayload:
    """Keep original capture lineage; no model/Zone/identity/evidence fields."""
    return {
        "schemaVersion": 1,
        "purpose": "REGION_CONFIGURATION",
        "sourceKind": kind,
        "frame": {
            "cameraExternalId": frame.camera_external_id,
            "sessionId": str(frame.session_id),
            "sequenceNumber": str(frame.sequence_number),
            "capturedAt": frame.captured_at_iso,
            "width": jpeg.width,
            "height": jpeg.height,
            "imageDataUrl": "data:image/jpeg;base64," + base64.b64encode(jpeg.data).decode("ascii"),
            "imageSha256": hashlib.sha256(jpeg.data).hexdigest(),
        },
    }
