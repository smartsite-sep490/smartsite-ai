"""Paced, newest-frame replay display; finite durable evaluation stays separate."""

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.live_latest_source import LiveLatestFrameSource, MediaSink
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource


def build_replay_preview_source(
    config: StreamConfig, media_sink: MediaSink
) -> LiveLatestFrameSource:
    if config.is_live:
        raise ValueError("Replay preview requires a finite recorded source")
    source = OpenCvFrameSource(config.model_copy(update={"pace_replay": True}))
    latest = LiveLatestFrameSource(source)
    latest.set_media_sink(media_sink)
    return latest
