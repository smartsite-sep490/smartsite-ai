import asyncio
from datetime import UTC, datetime

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.testing import FakeFrameSource
from smartsite_ai.ingestion.worker import CameraIngestionWorker, StreamWorker


def make_test_frame(stream_id: str, seq: int) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id=stream_id,
        session_id=f"session-{stream_id}",
        camera_external_id=f"ext-{stream_id}",
        captured_at=datetime.now(UTC),
        width=1280,
        height=720,
        sequence_number=seq,
        payload=f"payload-{seq}",
    )


@pytest.mark.anyio
async def test_stream_worker_normal_lifecycle() -> None:
    config = StreamConfig(
        stream_id="cam-1",
        camera_external_id="ext-cam-1",
        source_url="rtsp://admin:pass@192.168.1.10:554/ch0",
        max_queue_size=5,
    )
    frames = [make_test_frame("cam-1", i) for i in range(1, 4)]
    source = FakeFrameSource(source_id="cam-1", initial_frames=frames, delay_between_frames=0.01)

    worker = StreamWorker(config=config, source=source)
    await worker.start()

    # Read frames from worker queue
    frame1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    frame2 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    frame3 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)

    assert frame1.sequence_number == 1
    assert frame2.sequence_number == 2
    assert frame3.sequence_number == 3

    status = worker.snapshot()
    assert status.metrics.frames_enqueued >= 3
    assert status.metrics.frames_dequeued == 3
    assert status.metrics.frames_dropped == 0
    assert "pass" not in status.sanitized_url
    assert "admin" not in status.sanitized_url

    await worker.stop()
    assert source.is_closed is True
    assert worker.state == StreamState.STOPPED


@pytest.mark.anyio
async def test_stream_worker_reconnect_on_connect_failure() -> None:
    config = StreamConfig(
        stream_id="cam-retry",
        camera_external_id="ext-retry",
        source_url="rtsp://192.168.1.50/live",
        reconnect_initial_delay=0.05,
        reconnect_max_delay=0.2,
        reconnect_jitter=0.0,
    )
    frames = [make_test_frame("cam-retry", 1)]
    source = FakeFrameSource(
        source_id="cam-retry",
        initial_frames=frames,
        connect_failures_before_success=2,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()

    # Worker should retry twice, succeed on 3rd, and deliver frame 1
    frame = await asyncio.wait_for(worker.get_frame(), timeout=2.0)
    assert frame.sequence_number == 1

    status = worker.snapshot()
    assert status.metrics.connection_errors == 2
    assert status.metrics.reconnect_attempts >= 2
    assert source.connect_calls == 3

    await worker.stop()


@pytest.mark.anyio
async def test_stream_worker_reconnect_on_read_failure() -> None:
    config = StreamConfig(
        stream_id="cam-read-fail",
        camera_external_id="ext-fail",
        source_url="rtsp://10.0.0.5/stream",
        reconnect_initial_delay=0.05,
        reconnect_max_delay=0.2,
        reconnect_jitter=0.0,
    )
    frames_batch_1 = [make_test_frame("cam-read-fail", 1)]
    frames_batch_2 = [make_test_frame("cam-read-fail", 2)]

    source = FakeFrameSource(
        source_id="cam-read-fail",
        initial_frames=frames_batch_1,
        fail_read_once_after=1,
        secondary_frames=frames_batch_2,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()

    f1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    assert f1.sequence_number == 1

    # Second frame should arrive after read failure & reconnect
    f2 = await asyncio.wait_for(worker.get_frame(), timeout=2.0)
    assert f2.sequence_number == 2

    status = worker.snapshot()
    assert status.metrics.read_errors >= 1
    assert status.metrics.reconnect_attempts >= 1

    await worker.stop()


@pytest.mark.anyio
async def test_stream_fault_isolation() -> None:
    """A failing stream must not impact or terminate a healthy stream."""
    healthy_config = StreamConfig(
        stream_id="healthy-cam",
        camera_external_id="ext-healthy",
        source_url="rtsp://healthy/live",
        max_queue_size=10,
    )
    failing_config = StreamConfig(
        stream_id="failing-cam",
        camera_external_id="ext-failing",
        source_url="rtsp://failing/live",
        reconnect_initial_delay=0.05,
        reconnect_max_delay=0.1,
    )

    healthy_frames = [make_test_frame("healthy-cam", i) for i in range(1, 30)]
    healthy_source = FakeFrameSource(
        source_id="healthy-cam",
        initial_frames=healthy_frames,
        delay_between_frames=0.05,
    )
    failing_source = FakeFrameSource(
        source_id="failing-cam",
        initial_frames=[],
        permanent_connect_failure=True,
    )

    manager = CameraIngestionWorker()
    healthy_worker = manager.add_stream(healthy_config, source=healthy_source)
    manager.add_stream(failing_config, source=failing_source)

    await manager.start()

    # Healthy worker produces frames steadily
    f1 = await asyncio.wait_for(healthy_worker.get_frame(), timeout=1.0)
    f2 = await asyncio.wait_for(healthy_worker.get_frame(), timeout=1.0)
    assert f1.sequence_number == 1
    assert f2.sequence_number == 2

    # Failing worker reports backoff/errors but manager is still running
    await asyncio.sleep(0.2)
    snapshot = manager.snapshot()
    assert snapshot.status == "running"
    assert snapshot.stream_count == 2
    assert snapshot.streams["healthy-cam"].state == StreamState.STREAMING
    assert snapshot.streams["failing-cam"].state in (StreamState.BACKOFF, StreamState.CONNECTING)
    assert snapshot.streams["failing-cam"].metrics.connection_errors > 0

    await manager.stop()
    assert healthy_source.is_closed is True
    assert failing_source.is_closed is True


@pytest.mark.anyio
async def test_bounded_queue_backpressure_in_worker() -> None:
    config = StreamConfig(
        stream_id="cam-backpressure",
        camera_external_id="ext-bp",
        source_url="rtsp://camera/feed",
        max_queue_size=2,
    )
    frames = [make_test_frame("cam-backpressure", i) for i in range(1, 6)]
    source = FakeFrameSource(
        source_id="cam-backpressure",
        initial_frames=frames,
        delay_between_frames=0.01,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()

    # Wait for producer to push all 5 frames without consumer reading
    await asyncio.sleep(0.15)

    status = worker.snapshot()
    assert status.metrics.frames_enqueued == 5
    # Queue size is 2, so 3 frames must have been dropped
    assert status.metrics.frames_dropped == 3

    # Consumer now drains the queue: must be the two newest frames (4 and 5)
    f_first = await worker.get_frame()
    f_second = await worker.get_frame()
    assert f_first.sequence_number == 4
    assert f_second.sequence_number == 5

    await worker.stop()


@pytest.mark.anyio
async def test_camera_ingestion_worker_manager_lifecycle() -> None:
    manager = CameraIngestionWorker()
    assert manager.snapshot().status == "idle"

    config1 = StreamConfig(stream_id="c1", camera_external_id="ext-1", source_url="rtsp://c1")
    config2 = StreamConfig(stream_id="c2", camera_external_id="ext-2", source_url="rtsp://c2")

    s1 = FakeFrameSource("c1", [make_test_frame("c1", 1)])
    s2 = FakeFrameSource("c2", [make_test_frame("c2", 1)])

    manager.add_stream(config1, source=s1)
    manager.add_stream(config2, source=s2)

    assert manager.snapshot().stream_count == 2

    await manager.start()
    assert manager.snapshot().status == "running"

    await manager.stop()
    assert manager.snapshot().status == "stopped"
    assert s1.is_closed is True
    assert s2.is_closed is True


def test_import_side_effects_are_zero() -> None:
    # Verify module can be imported cleanly without creating running loops or background tasks
    import smartsite_ai.ingestion

    assert hasattr(smartsite_ai.ingestion, "CameraIngestionWorker")
    assert hasattr(smartsite_ai.ingestion, "FrameEnvelope")
