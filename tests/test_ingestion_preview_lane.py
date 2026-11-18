import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.queue import PreviewFrameQueue, QueueClosedError
from smartsite_ai.ingestion.testing import FakeFrameSource, make_test_frame
from smartsite_ai.ingestion.worker import StreamWorker


async def settled(predicate: Callable[[], bool]) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Capture did not reach the expected state")


@pytest.mark.anyio
async def test_empty_waiter_receives_first_frame_after_source_invalidation() -> None:
    queue = PreviewFrameQueue()
    pending = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    queue.clear()
    fresh = make_test_frame("preview", 0)
    queue.put(fresh)
    try:
        assert await asyncio.wait_for(pending, 0.2) is fresh
    finally:
        queue.close()


@pytest.mark.anyio
@pytest.mark.parametrize("terminal", [False, True])
async def test_pending_preview_get_cannot_return_frame_from_before_invalidation(
    terminal: bool,
) -> None:
    queue = PreviewFrameQueue()
    old, fresh = make_test_frame("preview", 0), make_test_frame("preview", 1)
    pending = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    queue.put(old)
    # Let the internal queue task acquire the frame while its outer waiter is suspended.
    await asyncio.sleep(0)
    if terminal:
        queue.close()
        with pytest.raises(QueueClosedError):
            await asyncio.wait_for(pending, 1)
    else:
        queue.clear()
        queue.put(fresh)
        assert await asyncio.wait_for(pending, 1) is fresh
    queue.close()


@pytest.mark.anyio
async def test_preview_keeps_latest_frame_without_waiting_for_inference() -> None:
    start = datetime.now(UTC)
    frames = [
        make_test_frame("preview", i, captured_at=start + timedelta(milliseconds=i * 10))
        for i in range(4)
    ]
    source = FakeFrameSource("preview", frames, block_when_exhausted=True)
    preview = PreviewFrameQueue()
    worker = StreamWorker(
        StreamConfig(
            stream_id="preview",
            camera_external_id="ext-preview",
            source_url="0",
            is_live=True,
            target_fps=1,
            max_queue_size=1,
        ),
        source=source,
        preview_frames=preview,
    )
    await worker.start()
    try:
        await settled(lambda: source.read_calls == 5)
        # A stalled inference reader neither blocks preview nor makes it open a source.
        assert source.connect_calls == 1
        assert preview.enqueued_count == 4
        assert preview.dropped_count == 3
        assert await preview.get() is frames[-1]
        assert await worker.get_frame() is frames[0]
        assert worker.owns_current_frame(frames[0])
        assert worker.sampled_out_frames == 3
    finally:
        await worker.stop()
    assert source.close_calls == 1
    assert not worker.owns_current_frame(frames[0])


@pytest.mark.anyio
async def test_preview_disposal_does_not_stop_capture_or_inference() -> None:
    frames = [make_test_frame("preview", i) for i in range(3)]
    source = FakeFrameSource("preview", frames, block_when_exhausted=True)
    preview = PreviewFrameQueue()
    worker = StreamWorker(
        StreamConfig(
            stream_id="preview",
            camera_external_id="ext-preview",
            source_url="0",
            is_live=True,
            max_queue_size=3,
        ),
        source=source,
        preview_frames=preview,
    )
    preview.close()
    await worker.start()
    try:
        await settled(lambda: source.read_calls == 4)
        assert source.connect_calls == 1
        assert source.close_calls == 0
        assert [await worker.get_frame() for _ in frames] == frames
    finally:
        await worker.stop()


@pytest.mark.anyio
async def test_source_failure_invalidates_queued_preview_pixels() -> None:
    source = FakeFrameSource("preview", [make_test_frame("preview", 0)])
    preview = PreviewFrameQueue()
    worker = StreamWorker(
        StreamConfig(
            stream_id="preview",
            camera_external_id="ext-preview",
            source_url="0",
            is_live=True,
            max_consecutive_failures=1,
        ),
        source=source,
        preview_frames=preview,
    )
    await worker.start()
    try:
        await settled(lambda: preview.is_closed)
        assert preview.empty()
        with pytest.raises(QueueClosedError):
            await preview.get()
    finally:
        await worker.stop()


@pytest.mark.anyio
async def test_invalid_identity_never_reaches_preview_and_stop_releases_waiters() -> None:
    source = FakeFrameSource(
        "preview",
        [make_test_frame("preview", 0, camera_external_id="another-camera")],
        block_when_exhausted=True,
    )
    preview = PreviewFrameQueue()
    worker = StreamWorker(
        StreamConfig(
            stream_id="preview",
            camera_external_id="ext-preview",
            source_url="0",
            is_live=True,
        ),
        source=source,
        preview_frames=preview,
    )
    await worker.start()
    await settled(lambda: source.read_calls == 2)
    assert preview.enqueued_count == 0
    pending = asyncio.create_task(preview.get())
    try:
        await worker.stop()
        with pytest.raises(QueueClosedError):
            await asyncio.wait_for(pending, 1)
    finally:
        await worker.stop()
