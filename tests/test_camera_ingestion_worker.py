import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import QueueClosedError
from smartsite_ai.ingestion.source import SourceConnectionError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.testing import (
    FakeBlockingSource,
    FakeFrameSource,
    FakeSleeper,
)
from smartsite_ai.ingestion.worker import CameraIngestionWorker, StreamWorker


def make_test_frame(
    stream_id: str,
    seq: int,
    session_id: str | None = None,
    camera_external_id: str | None = None,
    captured_at: datetime | None = None,
) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id=stream_id,
        session_id=session_id or f"session-{stream_id}",
        camera_external_id=camera_external_id or f"ext-{stream_id}",
        captured_at=captured_at or datetime.now(UTC),
        width=1280,
        height=720,
        sequence_number=seq,
        payload=f"payload-{seq}",
    )


async def wait_until(predicate: Callable[[], bool], max_iterations: int = 200) -> None:
    """Bounded cooperative polling using asyncio.sleep(0) to eliminate wall-clock flakiness."""
    for _ in range(max_iterations):
        if predicate():
            return
        await asyncio.sleep(0)
    raise TimeoutError("Predicate condition not met within bounded cooperative iterations")


@pytest.mark.anyio
async def test_stream_worker_normal_lifecycle() -> None:
    config = StreamConfig(
        stream_id="cam-1",
        camera_external_id="ext-cam-1",
        source_url="rtsp://admin:pass@192.168.1.10:554/ch0",
        max_queue_size=5,
        is_live=False,
    )
    frames = [make_test_frame("cam-1", i, camera_external_id="ext-cam-1") for i in range(1, 4)]
    source = FakeFrameSource(
        source_id="cam-1",
        initial_frames=frames,
        is_live=False,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
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
        assert status.metrics.integrity_errors == 0
        assert status.metrics.sequence_errors == 0
        assert status.metrics.sampled_out_frames == 0
        assert "pass" not in status.sanitized_url
        assert "admin" not in status.sanitized_url
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)

    assert source.close_calls == 1
    assert worker.state == StreamState.STOPPED


@pytest.mark.anyio
async def test_stream_worker_reconnect_on_connect_failure() -> None:
    config = StreamConfig(
        stream_id="cam-retry",
        camera_external_id="ext-retry",
        source_url="rtsp://192.168.1.50/live",
        reconnect_initial_delay=0.001,
        reconnect_max_delay=0.005,
        reconnect_jitter=0.0,
        is_live=False,
    )
    frames = [make_test_frame("cam-retry", 1, camera_external_id="ext-retry")]
    source = FakeFrameSource(
        source_id="cam-retry",
        initial_frames=frames,
        connect_failures_before_success=2,
        is_live=False,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        frame = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert frame.sequence_number == 1

        status = worker.snapshot()
        assert status.metrics.connection_errors == 2
        assert status.metrics.reconnect_attempts >= 2
        assert source.connect_calls == 3
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_reconnect_does_not_reset_backoff_on_immediate_failure() -> None:
    """If a connection immediately fails after connecting, backoff must NOT reset (Blocker 4)."""
    sleeper = FakeSleeper()
    config = StreamConfig(
        stream_id="cam-flapping",
        camera_external_id="ext-flapping",
        source_url="rtsp://10.0.0.99/live",
        reconnect_initial_delay=1.0,
        reconnect_max_delay=16.0,
        reconnect_backoff_factor=2.0,
        reconnect_jitter=0.0,
        min_stable_frames=5,
        max_consecutive_failures=3,
    )

    source = FakeFrameSource(
        source_id="cam-flapping",
        initial_frames=[make_test_frame("cam-flapping", 1, camera_external_id="ext-flapping")],
        fail_read_once_after=1,
    )

    worker = StreamWorker(config=config, source=source, sleeper=sleeper)
    await worker.start()
    try:
        await wait_until(lambda: len(sleeper.sleep_calls) >= 1)
        assert sleeper.sleep_calls[0] == pytest.approx(1.0)
        assert worker.consecutive_failures >= 1
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_resets_backoff_after_reaching_min_stable_frames() -> None:
    """Backoff is reset only after yielding min_stable_frames successfully (Blocker 4)."""
    config = StreamConfig(
        stream_id="cam-stable",
        camera_external_id="ext-stable",
        source_url="rtsp://10.0.0.99/live",
        min_stable_frames=3,
        reconnect_initial_delay=0.001,
        reconnect_jitter=0.0,
        is_live=False,
    )

    frames = [
        make_test_frame("cam-stable", i, camera_external_id="ext-stable") for i in range(1, 4)
    ]
    source = FakeFrameSource(source_id="cam-stable", initial_frames=frames, is_live=False)

    worker = StreamWorker(config=config, source=source)
    worker.consecutive_failures = 4
    await worker.start()
    try:
        await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert worker.consecutive_failures == 0
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_eof_handling_live_vs_finite() -> None:
    """Finite clip stops cleanly; live stream treats EOF as disconnect."""
    # 1. Finite clip
    finite_config = StreamConfig(
        stream_id="clip-1",
        camera_external_id="ext-clip-1",
        source_url="/var/data/video.mp4",
        is_live=False,
    )
    s1 = FakeFrameSource(
        "clip-1",
        [make_test_frame("clip-1", 1, camera_external_id="ext-clip-1")],
        is_live=False,
    )
    w1 = StreamWorker(config=finite_config, source=s1)
    await w1.start()
    try:
        f1 = await asyncio.wait_for(w1.get_frame(), timeout=1.0)
        assert f1.sequence_number == 1
        await wait_until(lambda: w1.state == StreamState.STOPPED)
        assert w1.state == StreamState.STOPPED
    finally:
        await asyncio.wait_for(w1.stop(), timeout=1.0)

    # 2. Live camera EOF (unexpected disconnect)
    sleeper = FakeSleeper()
    live_config = StreamConfig(
        stream_id="live-1",
        camera_external_id="ext-live-1",
        source_url="rtsp://cam/live",
        is_live=True,
        max_consecutive_failures=2,
    )
    s2 = FakeFrameSource(
        "live-1",
        [make_test_frame("live-1", 1, camera_external_id="ext-live-1")],
        is_live=True,
    )
    w2 = StreamWorker(config=live_config, source=s2, sleeper=sleeper)
    await w2.start()
    try:
        f2 = await asyncio.wait_for(w2.get_frame(), timeout=1.0)
        assert f2.sequence_number == 1
        await wait_until(lambda: w2.state in (StreamState.BACKOFF, StreamState.ERROR))
        assert w2.state in (StreamState.BACKOFF, StreamState.ERROR)
        assert w2.last_error == "connection_closed_eof"
    finally:
        await asyncio.wait_for(w2.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_frame_integrity_validation() -> None:
    """Frames with mismatched stream_id or camera_external_id must be rejected (Blocker 5)."""
    config = StreamConfig(
        stream_id="expected-stream",
        camera_external_id="expected-cam",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )

    bad_frame_stream = make_test_frame("wrong-stream", 1, camera_external_id="expected-cam")
    bad_frame_cam = make_test_frame("expected-stream", 2, camera_external_id="wrong-cam")
    good_frame = make_test_frame("expected-stream", 3, camera_external_id="expected-cam")

    source = FakeFrameSource(
        source_id="expected-stream",
        initial_frames=[bad_frame_stream, bad_frame_cam, good_frame],
        is_live=False,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        received = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert received.sequence_number == 3

        status = worker.snapshot()
        assert status.metrics.integrity_errors == 2
        assert "frame_integrity_error" in (status.metrics.last_error or "")
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_session_and_sequence_semantics() -> None:
    """Verify session consistency and monotonic sequence enforcement (MF05/MF06)."""
    config = StreamConfig(
        stream_id="seq-cam",
        camera_external_id="ext-seq",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )

    f1 = make_test_frame("seq-cam", 10, session_id="sess-A", camera_external_id="ext-seq")
    f2 = make_test_frame("seq-cam", 11, session_id="sess-A", camera_external_id="ext-seq")
    # Duplicate sequence: 11
    f3_dup = make_test_frame("seq-cam", 11, session_id="sess-A", camera_external_id="ext-seq")
    # Out of order sequence: 9 < 11
    f4_ooo = make_test_frame("seq-cam", 9, session_id="sess-A", camera_external_id="ext-seq")
    # Session switch mid-connection
    f5_bad_sess = make_test_frame("seq-cam", 12, session_id="sess-B", camera_external_id="ext-seq")
    # Valid next frame in same session: 12
    f6_valid = make_test_frame("seq-cam", 12, session_id="sess-A", camera_external_id="ext-seq")

    source = FakeFrameSource(
        source_id="seq-cam",
        initial_frames=[f1, f2, f3_dup, f4_ooo, f5_bad_sess, f6_valid],
        is_live=False,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        out1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert out1.sequence_number == 10

        out2 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert out2.sequence_number == 11

        out3 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert out3.sequence_number == 12

        status = worker.snapshot()
        # 3 violations: duplicate 11, out-of-order 9, session switch to sess-B
        assert status.metrics.sequence_errors == 3
        assert status.metrics.last_error == "session_sequence_error"
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_single_use_lifecycle_rejects_restart() -> None:
    """StreamWorker is single-use: attempting to start() after stop() must raise RuntimeError."""
    config = StreamConfig(
        stream_id="single-use-cam",
        camera_external_id="ext-su",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )
    source = FakeFrameSource("single-use-cam", [], is_live=False)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    await worker.stop()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await worker.start()


@pytest.mark.anyio
async def test_stream_worker_eof_natural_completion_lifecycle() -> None:
    """Finite source EOF must finalize worker lifecycle without explicit stop() call."""
    config = StreamConfig(
        stream_id="cam-eof",
        camera_external_id="ext-eof",
        source_url="rtsp://10.0.0.1/video.mp4",
        is_live=False,
    )
    frames = [
        make_test_frame("cam-eof", 1, camera_external_id="ext-eof"),
        make_test_frame("cam-eof", 2, camera_external_id="ext-eof"),
    ]
    source = FakeFrameSource(source_id="cam-eof", initial_frames=frames, is_live=False)
    worker = StreamWorker(config=config, source=source)

    await worker.start()

    f1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    f2 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
    assert f1.sequence_number == 1
    assert f2.sequence_number == 2

    # Wait for loop to naturally reach EOF and terminate
    await wait_until(lambda: worker.state == StreamState.STOPPED)

    # Invariant: Do NOT call worker.stop() before asserting terminal finalization
    assert worker.state == StreamState.STOPPED
    assert worker.stopped_at is not None
    assert worker.stopped_at.tzinfo is not None
    assert worker.queue.is_closed is True
    assert source.close_calls == 1

    # Queue must wake/reject subsequent gets with QueueClosedError
    with pytest.raises(QueueClosedError):
        await worker.get_frame()

    # Restart must be rejected after natural EOF
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await worker.start()

    # Subsequent stop() call is safe idempotent no-op and preserves stopped_at and close_calls
    initial_stopped_at = worker.stopped_at
    await worker.stop()
    assert source.close_calls == 1
    assert worker.state == StreamState.STOPPED
    assert worker.stopped_at == initial_stopped_at


@pytest.mark.anyio
async def test_stream_worker_error_terminal_completion_lifecycle() -> None:
    """Terminal error path (max_consecutive_failures) must finalize lifecycle without stop()."""
    config = StreamConfig(
        stream_id="cam-term-err",
        camera_external_id="ext-err",
        source_url="rtsp://10.0.0.1/live",
        reconnect_initial_delay=0.001,
        reconnect_max_delay=0.002,
        max_consecutive_failures=2,
    )
    source = FakeFrameSource(
        source_id="cam-term-err",
        initial_frames=[],
        permanent_connect_failure=True,
    )
    sleeper = FakeSleeper()
    worker = StreamWorker(config=config, source=source, sleeper=sleeper)

    await worker.start()

    # Wait for terminal ERROR state
    await wait_until(lambda: worker.state == StreamState.ERROR)

    # Invariant: Do NOT call worker.stop() before assertions
    assert worker.state == StreamState.ERROR
    assert worker.stopped_at is not None
    assert worker.queue.is_closed is True
    assert source.close_calls == 1

    with pytest.raises(QueueClosedError):
        await worker.get_frame()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await worker.start()

    # stop() preserves ERROR state, stopped_at, and close count
    err_stopped_at = worker.stopped_at
    await worker.stop()
    assert worker.state == StreamState.ERROR
    assert worker.stopped_at == err_stopped_at
    assert source.close_calls == 1


@pytest.mark.anyio
async def test_stream_worker_idempotent_start() -> None:
    """Calling start() on an already streaming worker is an idempotent no-op."""
    config = StreamConfig(
        stream_id="idemp-cam",
        camera_external_id="ext-idemp",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )
    frames = [make_test_frame("idemp-cam", 1, camera_external_id="ext-idemp")]
    source = FakeFrameSource("idemp-cam", frames, is_live=False)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    # Second start while already active
    await worker.start()
    try:
        f = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert f.sequence_number == 1
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_source_close_exactly_once() -> None:
    """Verify that source.close() is invoked exactly once during worker teardown."""
    config = StreamConfig(
        stream_id="close-once-cam",
        camera_external_id="ext-co",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )
    source = FakeFrameSource("close-once-cam", [], is_live=False)
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    await worker.stop()
    # Second stop call
    await worker.stop()

    assert source.close_calls == 1


@pytest.mark.anyio
async def test_stream_worker_deterministic_fps_sampling() -> None:
    """Worker deterministically drops frames that arrive faster than target_fps (Blocker 2)."""
    config = StreamConfig(
        stream_id="cam-sampling",
        camera_external_id="ext-sampling",
        source_url="rtsp://10.0.0.1/live",
        target_fps=2.0,  # 1 frame per 0.5s
        is_live=False,
    )

    t0 = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
    frames = [
        make_test_frame("cam-sampling", 1, camera_external_id="ext-sampling", captured_at=t0),
        make_test_frame(
            "cam-sampling",
            2,
            camera_external_id="ext-sampling",
            captured_at=t0 + timedelta(seconds=0.1),
        ),
        make_test_frame(
            "cam-sampling",
            3,
            camera_external_id="ext-sampling",
            captured_at=t0 + timedelta(seconds=0.2),
        ),
        make_test_frame(
            "cam-sampling",
            4,
            camera_external_id="ext-sampling",
            captured_at=t0 + timedelta(seconds=0.55),
        ),
        make_test_frame(
            "cam-sampling",
            5,
            camera_external_id="ext-sampling",
            captured_at=t0 + timedelta(seconds=0.65),
        ),
    ]

    source = FakeFrameSource(source_id="cam-sampling", initial_frames=frames, is_live=False)
    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        f1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert f1.sequence_number == 1

        f4 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert f4.sequence_number == 4

        await wait_until(lambda: worker.snapshot().metrics.sampled_out_frames == 3)
        status = worker.snapshot()
        assert status.metrics.sampled_out_frames == 3
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_hung_read_cancellation_with_blocking_source() -> None:
    """Stopping a worker while reading from a hung source must complete cleanly (Blocker 3)."""
    config = StreamConfig(
        stream_id="cam-hung",
        camera_external_id="ext-hung",
        source_url="rtsp://hung-camera/feed",
    )
    source = FakeBlockingSource(source_id="cam-hung")
    worker = StreamWorker(config=config, source=source)

    await worker.start()
    try:
        await wait_until(lambda: source.read_calls >= 1)
        assert source.read_calls >= 1
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)

    assert source.close_calls == 1
    assert worker.state == StreamState.STOPPED
    assert worker.queue.is_closed is True


@pytest.mark.anyio
async def test_stream_fault_isolation() -> None:
    """A failing stream must not impact or terminate a healthy stream."""
    healthy_config = StreamConfig(
        stream_id="healthy-cam",
        camera_external_id="ext-healthy",
        source_url="rtsp://healthy/live",
        max_queue_size=10,
        is_live=False,
    )
    failing_config = StreamConfig(
        stream_id="failing-cam",
        camera_external_id="ext-failing",
        source_url="rtsp://failing/live",
        reconnect_initial_delay=0.001,
        reconnect_max_delay=0.002,
        max_consecutive_failures=2,
    )

    healthy_frames = [
        make_test_frame("healthy-cam", i, camera_external_id="ext-healthy") for i in range(1, 10)
    ]
    healthy_source = FakeFrameSource(
        source_id="healthy-cam",
        initial_frames=healthy_frames,
        is_live=False,
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
    try:
        f1 = await asyncio.wait_for(healthy_worker.get_frame(), timeout=1.0)
        f2 = await asyncio.wait_for(healthy_worker.get_frame(), timeout=1.0)
        assert f1.sequence_number == 1
        assert f2.sequence_number == 2

        await wait_until(
            lambda: manager.snapshot().streams["failing-cam"].metrics.connection_errors > 0
        )
        snapshot = manager.snapshot()
        assert snapshot.status == "running"
        assert snapshot.stream_count == 2
        assert snapshot.streams["failing-cam"].metrics.connection_errors > 0
    finally:
        await asyncio.wait_for(manager.stop(), timeout=1.0)

    assert healthy_source.close_calls == 1
    assert failing_source.close_calls == 1


@pytest.mark.anyio
async def test_bounded_queue_backpressure_in_worker() -> None:
    config = StreamConfig(
        stream_id="cam-backpressure",
        camera_external_id="ext-bp",
        source_url="rtsp://camera/feed",
        max_queue_size=2,
        is_live=False,
    )
    frames = [
        make_test_frame("cam-backpressure", i, camera_external_id="ext-bp") for i in range(1, 6)
    ]
    source = FakeFrameSource(
        source_id="cam-backpressure",
        initial_frames=frames,
        is_live=False,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        await wait_until(lambda: worker.snapshot().metrics.frames_enqueued == 5)
        status = worker.snapshot()
        assert status.metrics.frames_enqueued == 5
        assert status.metrics.frames_dropped == 3

        f_first = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        f_second = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert f_first.sequence_number == 4
        assert f_second.sequence_number == 5
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_camera_ingestion_worker_manager_lifecycle_and_invariants() -> None:
    """Test manager invariants: duplicate rejection, immutability when running (Blocker 6 & 7)."""
    manager = CameraIngestionWorker()
    assert manager.snapshot().status == "idle"

    config1 = StreamConfig(
        stream_id="c1", camera_external_id="ext-1", source_url="rtsp://c1", is_live=False
    )
    config2 = StreamConfig(
        stream_id="c2", camera_external_id="ext-2", source_url="rtsp://c2", is_live=False
    )

    s1 = FakeFrameSource(
        "c1", [make_test_frame("c1", 1, camera_external_id="ext-1")], is_live=False
    )
    s2 = FakeFrameSource(
        "c2", [make_test_frame("c2", 1, camera_external_id="ext-2")], is_live=False
    )

    manager.add_stream(config1, source=s1)

    # Invariant 1: Duplicate stream_id must be rejected
    with pytest.raises(ValueError, match="already registered"):
        manager.add_stream(config1, source=s1)

    await manager.start()
    try:
        assert manager.snapshot().status == "running"

        # Invariant 2: Cannot add streams while manager is running (immutable once started)
        with pytest.raises(RuntimeError, match="Cannot add stream"):
            manager.add_stream(config2, source=s2)

        # Invariant 3: remove_stream stops the worker
        await manager.remove_stream("c1")
        assert manager.get_stream("c1") is None
        assert s1.close_calls == 1
    finally:
        await asyncio.wait_for(manager.stop(), timeout=1.0)

    assert manager.snapshot().status == "stopped"


@pytest.mark.anyio
async def test_camera_ingestion_worker_manager_restart_preserves_stopped_status() -> None:
    """Manager start -> stop -> start must raise RuntimeError and preserve 'stopped' status."""
    manager = CameraIngestionWorker()
    config = StreamConfig(
        stream_id="cam-mgr-restart",
        camera_external_id="ext-mgr-restart",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )
    source = FakeFrameSource(
        source_id="cam-mgr-restart",
        initial_frames=[
            make_test_frame("cam-mgr-restart", 1, camera_external_id="ext-mgr-restart")
        ],
        is_live=False,
    )
    manager.add_stream(config, source=source)

    assert manager.snapshot().status == "idle"

    await manager.start()
    assert manager.snapshot().status == "running"

    # Idempotent start while running
    await manager.start()
    assert manager.snapshot().status == "running"

    await manager.stop()
    assert manager.snapshot().status == "stopped"

    # Attempt to restart after stopped must raise BEFORE changing status
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await manager.start()

    # Snapshot and internal status must remain "stopped", NOT "running"
    snapshot = manager.snapshot()
    assert snapshot.status == "stopped"


@pytest.mark.anyio
async def test_camera_ingestion_worker_start_failure_does_not_leave_status_running() -> None:
    """If starting workers fails during manager.start(), status must not remain 'running'."""
    manager = CameraIngestionWorker()
    config = StreamConfig(
        stream_id="cam-fail-start",
        camera_external_id="ext-fs",
        source_url="rtsp://10.0.0.1/live",
        is_live=False,
    )
    source = FakeFrameSource(source_id="cam-fail-start", initial_frames=[], is_live=False)
    worker = manager.add_stream(config, source=source)

    # Manually stop the worker so worker.start() will raise RuntimeError
    await worker.stop()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await manager.start()

    # Status must NOT be running
    snapshot = manager.snapshot()
    assert snapshot.status == "stopped"


@pytest.mark.anyio
async def test_stream_worker_error_does_not_leak_raw_url_or_token() -> None:
    """StreamMetrics.last_error must never contain leaked passwords or tokens (Blocker 1)."""
    config = StreamConfig(
        stream_id="cam-secret-test",
        camera_external_id="ext-sec",
        source_url="rtsp://admin:super_secret_pw@10.0.0.1:554/live?token=secret_token_123",
        reconnect_initial_delay=0.001,
        max_consecutive_failures=2,
    )

    class ErrorRaisingSource(FakeFrameSource):
        async def connect(self) -> None:
            raise SourceConnectionError(
                "Failed to connect to rtsp://admin:super_secret_pw@10.0.0.1:554/live?token=secret_token_123"
            )

    source = ErrorRaisingSource(source_id="cam-secret-test")
    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        await wait_until(lambda: worker.snapshot().metrics.last_error is not None)
        status = worker.snapshot()
        assert status.metrics.last_error is not None
        assert "super_secret_pw" not in status.metrics.last_error
        assert "secret_token_123" not in status.metrics.last_error
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


@pytest.mark.anyio
async def test_stream_worker_sampling_resets_on_reconnect_with_lower_timestamp() -> None:
    """Frame sampling baseline must reset upon reconnect/new session even if timestamp is lower."""
    config = StreamConfig(
        stream_id="cam-resample",
        camera_external_id="ext-resample",
        source_url="rtsp://10.0.0.1/live",
        target_fps=1.0,
        reconnect_initial_delay=0.001,
        max_consecutive_failures=3,
        is_live=True,
    )

    t_session1 = datetime(2026, 9, 19, 12, 5, 0, tzinfo=UTC)
    t_session2 = datetime(2026, 9, 19, 12, 1, 0, tzinfo=UTC)

    f1 = make_test_frame(
        "cam-resample",
        1,
        session_id="sess-old",
        camera_external_id="ext-resample",
        captured_at=t_session1,
    )
    f2_lower = make_test_frame(
        "cam-resample",
        1,
        session_id="sess-new",
        camera_external_id="ext-resample",
        captured_at=t_session2,
    )

    source = FakeFrameSource(
        source_id="cam-resample",
        initial_frames=[f1],
        fail_read_once_after=1,
        secondary_frames=[f2_lower],
        is_live=True,
    )

    worker = StreamWorker(config=config, source=source)
    await worker.start()
    try:
        got1 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert got1.sequence_number == 1
        assert got1.session_id == "sess-old"

        # Frame with lower timestamp from new session must be accepted and not sampled out
        got2 = await asyncio.wait_for(worker.get_frame(), timeout=1.0)
        assert got2.sequence_number == 1
        assert got2.session_id == "sess-new"
        assert got2.captured_at == t_session2

        status = worker.snapshot()
        assert status.metrics.sampled_out_frames == 0
    finally:
        await asyncio.wait_for(worker.stop(), timeout=1.0)


def test_import_side_effects_are_zero() -> None:
    import smartsite_ai.ingestion

    assert hasattr(smartsite_ai.ingestion, "CameraIngestionWorker")
    assert hasattr(smartsite_ai.ingestion, "FrameEnvelope")
    assert hasattr(smartsite_ai.ingestion, "QueueClosedError")
    assert hasattr(smartsite_ai.ingestion, "FrameIntegrityError")
    assert hasattr(smartsite_ai.ingestion, "SessionSequenceError")
