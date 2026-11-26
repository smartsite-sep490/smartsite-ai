"""Behavior shared by accelerated and fallback preview encoders."""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from PIL import Image

from smartsite_ai import realtime
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.runtime.media_preview import encode_media_frame


def test_fast_jpeg_preview_helper_and_pil_fallback_produce_exact_bgr_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from smartsite_ai.runtime import jpeg_preview

    small = FrameEnvelope(
        stream_id="cam-1",
        session_id=uuid4(),
        camera_external_id="CAM-1",
        captured_at=datetime.now(UTC),
        width=4,
        height=2,
        sequence_number=1,
        payload=bytes([0, 0, 255]) * 4 * 2,
    )
    large = FrameEnvelope(
        stream_id="cam-1",
        session_id=small.session_id,
        camera_external_id="CAM-1",
        captured_at=small.captured_at,
        width=1920,
        height=1080,
        sequence_number=2,
        payload=bytes([10, 100, 200]) * (1920 * 1080),
    )

    for disable_cv2 in (False, True):
        if disable_cv2:
            monkeypatch.setattr(jpeg_preview, "_cv2_module", lambda: None)
        p_small = realtime.build_realtime_preview(None, small)
        raw_small = base64.b64decode(p_small["imageDataUrl"].split(",")[1])
        img_small = Image.open(io.BytesIO(raw_small))
        assert img_small.size == (4, 2)
        assert img_small.getpixel((0, 0))[0] > 235
        assert img_small.getpixel((0, 0))[2] < 15

        p_large = realtime.build_realtime_preview(None, large)
        assert (p_large["width"], p_large["height"]) == (1280, 720)
        decoded_large = base64.b64decode(p_large["imageDataUrl"].split(",")[1])
        assert Image.open(io.BytesIO(decoded_large)).size == (1280, 720)
        m_large = encode_media_frame(large)
        assert (m_large["width"], m_large["height"]) == (640, 360)
