from datetime import UTC, datetime

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import BoundedFrameQueue


def make_frame(seq: int) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="s1",
        session_id="sess1",
        camera_external_id="cam1",
        captured_at=datetime.now(UTC),
        width=640,
        height=480,
        sequence_number=seq,
        payload=f"frame-{seq}",
    )


@pytest.mark.anyio
async def test_bounded_queue_enqueue_and_dequeue_without_drops() -> None:
    queue = BoundedFrameQueue(maxsize=3)
    f1 = make_frame(1)
    f2 = make_frame(2)

    queue.put(f1)
    queue.put(f2)

    assert queue.qsize() == 2
    assert queue.dropped_count == 0
    assert queue.enqueued_count == 2
    assert queue.dequeued_count == 0

    out1 = await queue.get()
    out2 = await queue.get()

    assert out1.sequence_number == 1
    assert out2.sequence_number == 2
    assert queue.dequeued_count == 2
    assert queue.qsize() == 0


@pytest.mark.anyio
async def test_bounded_queue_drop_stale_policy() -> None:
    queue = BoundedFrameQueue(maxsize=2)
    f1 = make_frame(1)
    f2 = make_frame(2)
    f3 = make_frame(3)
    f4 = make_frame(4)

    queue.put(f1)
    queue.put(f2)
    assert queue.qsize() == 2
    assert queue.dropped_count == 0

    # Putting f3 drops f1
    dropped_frame = queue.put(f3)
    assert dropped_frame is not None
    assert dropped_frame.sequence_number == 1
    assert queue.dropped_count == 1
    assert queue.enqueued_count == 3
    assert queue.qsize() == 2

    # Putting f4 drops f2
    dropped_frame = queue.put(f4)
    assert dropped_frame is not None
    assert dropped_frame.sequence_number == 2
    assert queue.dropped_count == 2
    assert queue.enqueued_count == 4
    assert queue.qsize() == 2

    # Consumer gets newest frames: f3, then f4
    got1 = await queue.get()
    got2 = await queue.get()
    assert got1.sequence_number == 3
    assert got2.sequence_number == 4
    assert queue.dequeued_count == 2
    assert queue.qsize() == 0


def test_bounded_queue_invalid_maxsize() -> None:
    with pytest.raises(ValueError, match="maxsize must be at least 1"):
        BoundedFrameQueue(maxsize=0)
    with pytest.raises(ValueError, match="maxsize must be at least 1"):
        BoundedFrameQueue(maxsize=-5)


def test_bounded_queue_clear() -> None:
    queue = BoundedFrameQueue(maxsize=3)
    queue.put(make_frame(1))
    queue.put(make_frame(2))
    assert queue.qsize() == 2

    queue.clear()
    assert queue.qsize() == 0
    assert queue.empty()
    # counters are preserved
    assert queue.enqueued_count == 2
