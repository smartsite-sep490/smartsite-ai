from pathlib import Path

import pytest
from test_multi_camera_runtime import CAMERA_A, _document, _entry

from smartsite_ai.ingestion.queue import BoundedFrameQueue
from smartsite_ai.ingestion.testing import make_test_frame
from smartsite_ai.runtime.manifest import bind_manifest, parse_manifest


@pytest.mark.anyio
async def test_live_runtime_consumer_gets_latest_sample_after_slow_processing(tmp_path: Path):
    entry = _entry(
        "live",
        CAMERA_A,
        "CAM-A",
        source={"kind": "env", "name": "SMARTSITE_AI_LIVE_CAMERA"},
        outbox=str(tmp_path / "live.sqlite"),
    )
    entry.update(live=True, targetFps=1)
    bound = bind_manifest(
        parse_manifest(_document(tmp_path, [entry])), {"SMARTSITE_AI_LIVE_CAMERA": "0"}
    )
    queue = BoundedFrameQueue(bound.cameras[0].source_config.max_queue_size)
    for sequence in range(4):
        queue.put(make_test_frame("live", sequence, camera_external_id="CAM-A"))
    assert (await queue.get()).sequence_number == 3
    assert queue.dropped_count == 3


def test_recorded_runtime_keeps_existing_bounded_replay_queue(tmp_path: Path):
    entry = _entry(
        "replay",
        CAMERA_A,
        "CAM-A",
        source={"kind": "env", "name": "SMARTSITE_AI_VIDEO"},
        outbox=str(tmp_path / "replay.sqlite"),
    )
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"synthetic-file-not-decoded")
    bound = bind_manifest(
        parse_manifest(_document(tmp_path, [entry])), {"SMARTSITE_AI_VIDEO": str(video)}
    )
    assert bound.cameras[0].source_config.max_queue_size == 5
