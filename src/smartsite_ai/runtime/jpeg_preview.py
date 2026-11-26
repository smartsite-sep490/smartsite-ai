"""Shared bounded BGR24-to-JPEG preview encoder using OpenCV with PIL fallback."""

from __future__ import annotations

import base64
import io
from typing import Any

from smartsite_ai.ingestion.envelope import FrameEnvelope


def _cv2_module() -> Any | None:
    try:
        import cv2
        import numpy
    except ImportError:
        return None
    if not hasattr(cv2, "resize") or not hasattr(cv2, "imencode"):
        return None
    return (cv2, numpy)


def thumbnail_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Compute aspect-preserving integer thumbnail dimensions bounded by ``max_side``."""
    if width <= max_side and height <= max_side:
        return width, height
    if width >= height:
        return max_side, max(1, round(height * max_side / width))
    return max(1, round(width * max_side / height)), max_side


def encode_bgr24_jpeg_data_url(
    frame: FrameEnvelope,
    *,
    max_side: int,
    qualities: tuple[int, ...],
    max_raw_bytes: int | None = None,
    max_data_url_bytes: int | None = None,
    error_message: str,
) -> tuple[str, int, int]:
    """Encode packed BGR24 pixels into a bounded JPEG data URL without swapping callers."""
    target_width, target_height = thumbnail_dimensions(frame.width, frame.height, max_side)
    cv_pair = _cv2_module()
    if cv_pair is not None:
        cv2, np = cv_pair
        bgr = np.frombuffer(frame.buffer, dtype=np.uint8).reshape((frame.height, frame.width, 3))
        if (target_width, target_height) != (frame.width, frame.height):
            bgr = cv2.resize(
                bgr,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        for quality in qualities:
            ok, encoded_buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            if not ok:
                raise ValueError(error_message)
            encoded = encoded_buf.tobytes()
            if max_raw_bytes is not None and len(encoded) > max_raw_bytes:
                continue
            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
            if max_data_url_bytes is not None and len(data_url) > max_data_url_bytes:
                continue
            return data_url, target_width, target_height
        raise ValueError(error_message)

    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
    if (target_width, target_height) != (frame.width, frame.height):
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    for quality in qualities:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
        encoded = output.getvalue()
        if max_raw_bytes is not None and len(encoded) > max_raw_bytes:
            continue
        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        if max_data_url_bytes is not None and len(data_url) > max_data_url_bytes:
            continue
        return data_url, image.width, image.height
    raise ValueError(error_message)
