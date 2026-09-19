"""Source protocol and credential-safe URL sanitization for camera ingestion."""

import re
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from smartsite_ai.ingestion.envelope import FrameEnvelope

# Matches URLs with user/password credentials (e.g., rtsp://user:pass@host:554/path)
_CREDENTIAL_URL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+-.]*://)([^:@/]+)(?::([^@/]+))?@")


def sanitize_stream_url(url: str) -> str:
    """Sanitize stream URL to ensure passwords and usernames are never logged or displayed.

    Examples:
        rtsp://admin:secret@10.0.0.1:554/live -> rtsp://***:***@10.0.0.1:554/live
        rtsp://viewer@10.0.0.1:554/live -> rtsp://***@10.0.0.1:554/live
        rtsp://10.0.0.1:554/live -> rtsp://10.0.0.1:554/live
        /dev/video0 -> /dev/video0
    """
    if not url or "://" not in url:
        return url

    try:
        parsed = urlsplit(url)
        if not parsed.username and not parsed.password:
            return url

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"

        if parsed.username and parsed.password:
            masked_netloc = f"***:***@{host}"
        elif parsed.username:
            masked_netloc = f"***@{host}"
        else:
            masked_netloc = f":***@{host}"

        return urlunsplit(
            (parsed.scheme, masked_netloc, parsed.path, parsed.query, parsed.fragment)
        )
    except Exception:
        # Fallback regex redaction if URL parsing encounters non-standard formats
        return _CREDENTIAL_URL_RE.sub(r"\1***:***@", url)


def sanitize_message_credentials(message: str) -> str:
    """Sanitize any URL credentials embedded in a log or exception message."""
    return _CREDENTIAL_URL_RE.sub(
        lambda m: f"{m.group(1)}***:***@" if m.group(3) else f"{m.group(1)}***@",
        message,
    )


class IngestionError(Exception):
    """Base exception for camera ingestion errors."""

    def __init__(self, message: str) -> None:
        sanitized = sanitize_message_credentials(str(message))
        super().__init__(sanitized)


class SourceConnectionError(IngestionError):
    """Raised when connection to camera or video stream fails."""


class SourceReadError(IngestionError):
    """Raised when reading a frame from connected camera fails."""


class SourceAuthenticationError(SourceConnectionError):
    """Raised when stream authentication fails (401 / 403 / auth timeout)."""


class SourceTimeoutError(IngestionError):
    """Raised when a camera stream operation times out."""


@runtime_checkable
class FrameSource(Protocol):
    """Protocol representing a video stream source (RTSP, video file, or fake).

    Implementations must not depend on YOLO, Ultralytics, or OpenCV.
    """

    @property
    def source_id(self) -> str:
        """Return unique source identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return true if source is currently connected."""
        ...

    async def connect(self) -> None:
        """Establish connection to stream source.

        Raises:
            SourceConnectionError: If connection fails.
            SourceAuthenticationError: If authentication fails.
        """
        ...

    async def read_frame(self) -> FrameEnvelope | None:
        """Read the next available frame envelope.

        Returns:
            FrameEnvelope if a frame was read, or None if stream has reached EOF or closed.

        Raises:
            SourceReadError: If frame capture or decoding fails.
            SourceTimeoutError: If read times out.
        """
        ...

    async def close(self) -> None:
        """Close source and release all underlying resources."""
        ...
