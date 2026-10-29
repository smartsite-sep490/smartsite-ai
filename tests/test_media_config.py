import pytest
from pydantic import ValidationError

from smartsite_ai.config import Settings


def test_raw_media_without_existing_latest_capture_is_rejected_at_configuration():
    with pytest.raises(ValidationError, match="raw LIVE media requires latest-frame capture"):
        Settings(_env_file=None, realtime_live_media_enabled=True, realtime_live_latest_frame=False)


def test_raw_media_requires_explicit_opt_in_and_accepts_same_capture_fanout():
    assert not Settings(_env_file=None).realtime_live_media_enabled
    configured = Settings(
        _env_file=None, realtime_live_media_enabled=True, realtime_live_latest_frame=True
    )
    assert configured.realtime_live_media_enabled
