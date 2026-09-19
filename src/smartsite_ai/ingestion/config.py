"""Stream configuration with credential-safe representation."""

from pydantic import BaseModel, ConfigDict, Field

from smartsite_ai.ingestion.source import sanitize_stream_url


class StreamConfig(BaseModel):
    """Configuration for a single camera or video ingestion stream."""

    model_config = ConfigDict(frozen=True)

    stream_id: str = Field(..., description="Unique stream identifier within the ingestion worker")
    camera_external_id: str = Field(..., description="Associated camera external ID")
    source_url: str = Field(..., description="Connection URL or path (e.g. RTSP, HTTP, file path)")
    max_queue_size: int = Field(
        default=5,
        ge=1,
        description="Capacity of bounded latest-frame queue",
    )
    target_fps: float | None = Field(
        default=None,
        gt=0,
        description="Optional target frame rate limit",
    )
    reconnect_initial_delay: float = Field(
        default=1.0,
        ge=0.01,
        description="Initial reconnect backoff in seconds",
    )
    reconnect_max_delay: float = Field(
        default=30.0,
        ge=0.05,
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

    @property
    def sanitized_source_url(self) -> str:
        """Return the stream URL with any credentials securely masked."""
        return sanitize_stream_url(self.source_url)

    def get_raw_source_url(self) -> str:
        """Return the unmasked source URL for actual connection establishing."""
        return self.source_url

    def __repr__(self) -> str:
        return (
            f"StreamConfig(stream_id='{self.stream_id}', "
            f"camera_external_id='{self.camera_external_id}', "
            f"source_url='{self.sanitized_source_url}', "
            f"max_queue_size={self.max_queue_size})"
        )

    def __str__(self) -> str:
        return self.__repr__()
