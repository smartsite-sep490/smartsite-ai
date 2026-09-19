"""Source protocol, credential-safe URL sanitization, and fail-closed error classification."""

import re
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from smartsite_ai.ingestion.envelope import FrameEnvelope

# Matches scheme and full authority containing one or more '@' characters.
# In RFC 3986, userinfo precedes the LAST '@' in the authority before host:port.
_AUTHORITY_URL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+-.]*://)([^/\s?#]+@[^/\s?#]+)")

# Allowlist of known-safe, non-credential camera query parameters (fail-closed design)
_SAFE_QUERY_KEYS = frozenset(
    {
        "channel",
        "ch",
        "subtype",
        "stream",
        "width",
        "height",
        "fps",
        "proto",
        "resolution",
        "transport",
    }
)


def _sanitize_authority(match: re.Match[str]) -> str:
    scheme = match.group(1)
    authority = match.group(2)
    # The last '@' in authority separates userinfo from host:port
    userinfo, at, hostport = authority.rpartition("@")
    if not at:
        return match.group(0)
    masked = "***:***" if ":" in userinfo else "***"
    return f"{scheme}{masked}@{hostport}"


def sanitize_stream_url(url: str) -> str:
    """Sanitize stream URL to ensure credentials, tokens, and fragments are never exposed.

    Enforces fail-closed sanitization:
    - Userinfo (username/password) is always masked, even if password contains '@' characters.
    - URL fragments (#...) are completely stripped.
    - Query parameters are fail-closed: only explicitly allowlisted benign keys keep values;
      all other values (tokens, secrets, signatures, unknown keys) are masked to '***'.

    Examples:
        rtsp://admin:secret@10.0.0.1:554/live -> rtsp://***:***@10.0.0.1:554/live
        rtsp://operator:p@ssword123@10.0.1.25/live -> rtsp://***:***@10.0.1.25/live
        http://camera/live?token=secret123&ch=1 -> http://camera/live?token=***&ch=1
        http://camera/live#access_token=secret -> http://camera/live
    """
    if not url or "://" not in url:
        return url

    # 1. Strip fragment completely before parsing
    clean_url = url.split("#")[0]

    # 2. Mask authority (handles passwords with '@')
    masked_url = _AUTHORITY_URL_RE.sub(_sanitize_authority, clean_url)

    # 3. Fail-closed query sanitization: only allowlist keeps value
    try:
        parsed = urlsplit(masked_url)
        masked_query = ""
        if parsed.query:
            query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
            sanitized_pairs = []
            for k, v in query_pairs:
                if k.lower() in _SAFE_QUERY_KEYS:
                    sanitized_pairs.append((k, v))
                else:
                    sanitized_pairs.append((k, "***"))
            masked_query = urlencode(sanitized_pairs, safe="*")

        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, masked_query, ""))
    except Exception:
        # Fallback query parameter masking
        return re.sub(
            r"([?&][^=&#\s]+)=([^&#\s]+)",
            lambda m: (
                m.group(0)
                if m.group(1).lstrip("?&").lower() in _SAFE_QUERY_KEYS
                else f"{m.group(1)}=***"
            ),
            masked_url,
        )


def sanitize_message_credentials(message: str) -> str:
    """Sanitize URL credentials (including passwords with '@') or tokens in messages."""
    sanitized = _AUTHORITY_URL_RE.sub(_sanitize_authority, message)
    # Strip any fragments in embedded URLs
    sanitized = re.sub(
        r"(https?|rtsp|rtsps)://[^\s#]+#[^\s]+", lambda m: m.group(0).split("#")[0], sanitized
    )
    # Mask query parameters with non-allowlisted keys
    return re.sub(
        r"([?&][a-zA-Z0-9_.-]+)=([^&#\s]+)",
        lambda m: (
            m.group(0)
            if m.group(1).lstrip("?&").lower() in _SAFE_QUERY_KEYS
            else f"{m.group(1)}=***"
        ),
        sanitized,
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


class FrameIntegrityError(IngestionError):
    """Raised when an ingested frame violates stream or camera identity invariants."""


class SessionSequenceError(IngestionError):
    """Raised when an ingested frame violates monotonic sequence or session invariants."""


def classify_error_reason(exc: BaseException) -> str:
    """Classify an exception into a safe, bounded reason string without arbitrary messages."""
    if isinstance(exc, SourceAuthenticationError):
        return "authentication_failed"
    if isinstance(exc, SourceTimeoutError):
        return "timeout"
    if isinstance(exc, SessionSequenceError):
        return "session_sequence_error"
    if isinstance(exc, FrameIntegrityError):
        return "frame_integrity_error"
    if isinstance(exc, SourceReadError):
        return "read_error"
    if isinstance(exc, SourceConnectionError):
        return "connection_error"
    if isinstance(exc, IngestionError):
        return "ingestion_error"

    # For unknown/runtime errors: return only type name and fixed safe label; never echo message
    return f"{type(exc).__name__}: unclassified_error"


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
