"""Synthetic JPEG budgets and exact source-purpose payload, without camera/model I/O."""

import base64
import hashlib
import importlib
import io

import pytest
from PIL import Image

from tests.configuration_capture_fakes import frame


def test_encoder_produces_decodable_jpeg_with_matching_hash_and_original_lineage():
    image_module = importlib.import_module("smartsite_ai.configuration_image")
    raw = frame(9223372036854775807, width=1600, height=800, payload=b"\x00\x20\x40" * 1600 * 800)
    jpeg = image_module.encode_configuration_image(raw)
    payload = image_module.configuration_payload(raw, "REPLAY", jpeg)
    encoded = base64.b64decode(payload["frame"]["imageDataUrl"].split(",", 1)[1], validate=True)
    decoded = Image.open(io.BytesIO(encoded))
    decoded.load()
    assert decoded.format == "JPEG" and decoded.size == (1280, 640)
    assert decoded.getpixel((0, 0)) == pytest.approx((64, 32, 0), abs=3)
    assert payload["frame"]["imageSha256"] == hashlib.sha256(encoded).hexdigest()
    assert payload["frame"]["sequenceNumber"] == "9223372036854775807"
    assert set(payload["frame"]) == {
        "cameraExternalId",
        "sessionId",
        "sequenceNumber",
        "capturedAt",
        "width",
        "height",
        "imageDataUrl",
        "imageSha256",
    }


@pytest.mark.parametrize("size", [(63, 64), (64, 63), (64, 2000)])
def test_small_or_extreme_aspect_ratio_frames_are_not_upscaled_or_misrepresented(size):
    module = importlib.import_module("smartsite_ai.configuration_image")
    with pytest.raises(ValueError, match="budget"):
        module.encode_configuration_image(
            frame(width=size[0], height=size[1], payload=b"\0" * (size[0] * size[1] * 3))
        )


@pytest.mark.parametrize(
    "data", [b"not-jpeg", b"\xff\xd8\xff\xd9" * (262144 + 1)], ids=["wrong-media", "oversize"]
)
def test_encoder_result_rejects_wrong_media_and_size(data):
    module = importlib.import_module("smartsite_ai.configuration_image")
    with pytest.raises(ValueError, match="budget"):
        module.ConfigurationJpeg(data, 64, 64)
