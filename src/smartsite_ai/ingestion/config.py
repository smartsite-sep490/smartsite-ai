"""Stream configuration with credential-safe representation and strict URL scheme validation."""

import re
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from smartsite_ai.ingestion.source import sanitize_stream_url

_ALLOWED_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https", "fake", "mock", "test"})
_ALLOWED_FILE_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
_DEVICE_INDEX_RE = re.compile(r"^(?:/dev/video\d+|\d+)$")


def _is_absolute_video_path(path_str: str) -> bool:
    """Check if path is an absolute video file path on POSIX or Windows."""
    if not path_str or ".." in path_str:
        return False

    lower = path_str.lower()
    if not lower.endswith(_ALLOWED_FILE_EXTENSIONS):
        return False

    # Disallow explicit file:// scheme
    if lower.startswith("file://"):
        return False

    # POSIX absolute path (starts with /)
    if path_str.startswith("/"):
        return True

    # Windows drive absolute path (e.g. C:\path or C:/path)
    if re.match(r"^[a-zA-Z]:[\\/]", path_str):
        return True

    # Windows UNC absolute path (e.g. \\server\share)
    return path_str.startswith("\\\\")


def _is_valid_source_url(raw_url: str) -> bool:
    """Validate whether source URL or device path is permitted for ingestion.

    Policy:
    - Supported network schemes: rtsp://, rtsps://, http://, https://, and test fixtures.
    - Device paths: Numeric index ('0', '1') or standard device nodes ('/dev/video0').
    - Local video files: Must be an absolute path ending in .mp4/.avi/.mkv/.mov.
      Relative paths, path traversal ('..'), and 'file://' URLs are strictly rejected.
    """
    if not raw_url or not raw_url.strip():
        return False

    raw_clean = raw_url.strip()

    # Reject any path traversal attempt immediately
    if ".." in raw_clean:
        return False

    # 1. Device index or device path (e.g. "0", "1", "/dev/video0")
    if _DEVICE_INDEX_RE.match(raw_clean):
        return True

    # 2. Local video file ending with supported extensions (must be absolute)
    if _is_absolute_video_path(raw_clean):
        return True

    # 3. URL with network scheme
    if "://" in raw_clean:
        try:
            parsed = urlsplit(raw_clean)
            scheme = parsed.scheme.lower()
            is_test_scheme = scheme in {"fake", "mock", "test"}
            if scheme in _ALLOWED_SCHEMES and (parsed.hostname or is_test_scheme):
                return True
        except Exception:
            return False

    return False


class StreamConfig(BaseModel):
    """Configuration for a single camera or video ingestion stream."""

    model_config = ConfigDict(
        frozen=True,
        hide_input_in_errors=True,
    )

    stream_id: str = Field(..., description="Unique stream identifier within the ingestion worker")
    camera_external_id: str = Field(..., description="Associated camera external ID")
    source_url: SecretStr = Field(
        ...,
        description="Credential-safe connection URL or path (e.g. RTSP, HTTP, file path)",
    )
    max_queue_size: int = Field(
        default=5,
        ge=1,
        description="Capacity of bounded latest-frame queue",
    )
    target_fps: float | None = Field(
        default=None,
        gt=0,
        description="Optional target frame rate limit for deterministic sampling",
    )
    is_live: bool = Field(
        default=True,
        description="Whether stream is ongoing live camera (True) or finite clip (False)",
    )
    min_stable_frames: int = Field(
        default=5,
        ge=1,
        description="Consecutive successful frames before resetting reconnect backoff counter",
    )
    reconnect_initial_delay: float = Field(
        default=1.0,
        ge=0.001,
        description="Initial reconnect backoff in seconds",
    )
    reconnect_max_delay: float = Field(
        default=30.0,
        ge=0.001,
        description="Maximum reconnect backoff in seconds",
    )
    reconnect_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier",
    )
    reconnect_jitter: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Jitter fraction applied to backoff",
    )
    max_consecutive_failures: int | None = Field(
        default=None,
        ge=1,
        description="Optional limit on consecutive failures before transitioning to error state",
    )

    @model_validator(mode="before")
    @classmethod
    def pre_mask_secrets(cls, data: object) -> object:
        """Ensure source_url is converted to SecretStr before field validation pipeline."""
        if isinstance(data, dict) and "source_url" in data:
            val = data["source_url"]
            if isinstance(val, str):
                data["source_url"] = SecretStr(val)
        return data

    @field_validator("source_url", mode="after")
    @classmethod
    def validate_source_url(cls, v: SecretStr) -> SecretStr:
        raw_val = v.get_secret_value()
        if not _is_valid_source_url(raw_val):
            # Never interpolate raw or partial secrets into the exception message
            raise ValueError(
                "Invalid stream source_url: must be a supported scheme "
                "(rtsp/rtsps/http/https), absolute video file (.mp4/.avi/.mkv/.mov), "
                "or camera device"
            )
        return v

    @property
    def sanitized_source_url(self) -> str:
        """Return the stream URL with any credentials securely masked."""
        return sanitize_stream_url(self.source_url.get_secret_value())

    def get_raw_source_url(self) -> str:
        """Return the unmasked source URL for actual connection establishing."""
        return self.source_url.get_secret_value()

    def __repr__(self) -> str:
        return (
            f"StreamConfig(stream_id='{self.stream_id}', "
            f"camera_external_id='{self.camera_external_id}', "
            f"source_url='{self.sanitized_source_url}', "
            f"max_queue_size={self.max_queue_size}, "
            f"is_live={self.is_live})"
        )

    def __str__(self) -> str:
        return self.__repr__()
