import pytest

import smartsite_ai.replay_preview_source as module
from smartsite_ai.ingestion.config import StreamConfig


def test_replay_preview_paces_and_uses_one_existing_capture(monkeypatch):
    constructed = []

    def source(config):
        constructed.append(config)
        return config

    monkeypatch.setattr(module, "OpenCvFrameSource", source)
    config = StreamConfig(
        stream_id="preview", camera_external_id="camera",
        source_url="D:/clips/replay.mp4", is_live=False,
    )
    sink = object()
    result = module.build_replay_preview_source(config, sink)
    assert len(constructed) == 1
    assert constructed[0].pace_replay
    assert not config.pace_replay
    assert result.source is constructed[0]
    assert result._media_sink is sink


def test_replay_preview_refuses_physical_camera():
    config = StreamConfig(
        stream_id="preview", camera_external_id="camera", source_url="0", is_live=True,
    )
    with pytest.raises(ValueError, match="finite recorded"):
        module.build_replay_preview_source(config, object())
