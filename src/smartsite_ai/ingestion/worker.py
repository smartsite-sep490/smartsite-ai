"""Camera ingestion worker and per-stream worker lifecycle."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

from smartsite_ai.ingestion.backoff import ExponentialBackoff
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import BoundedFrameQueue
from smartsite_ai.ingestion.source import (
    FrameIntegrityError,
    FrameSource,
    SessionSequenceError,
    SourceConnectionError,
    SourceReadError,
    classify_error_reason,
)
from smartsite_ai.ingestion.status import (
    StreamMetrics,
    StreamState,
    StreamStatus,
    WorkerStatus,
)


class StreamWorker:
    """Manages lifecycle, reconnection, backoff, and ingestion loop for a camera stream."""

    def __init__(
        self,
        config: StreamConfig,
        source: FrameSource | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.queue = BoundedFrameQueue(maxsize=config.max_queue_size)
        self.backoff = ExponentialBackoff(
            initial_delay=config.reconnect_initial_delay,
            max_delay=config.reconnect_max_delay,
            factor=config.reconnect_backoff_factor,
            jitter=config.reconnect_jitter,
        )
        self.state = StreamState.INITIALIZING
        self._sleeper = sleeper
        self._clock = clock or (lambda: datetime.now(UTC))

        self.connection_errors = 0
        self.read_errors = 0
        self.integrity_errors = 0
        self.sequence_errors = 0
        self.sampled_out_frames = 0
        self.reconnect_attempts = 0
        self.consecutive_failures = 0
        self._consecutive_successful_frames = 0

        self._current_session_id: str | None = None
        self._last_sequence_number: int | None = None

        self.last_frame_timestamp: datetime | None = None
        self._last_sampled_timestamp: datetime | None = None
        self.last_error: str | None = None
        self.started_at: datetime | None = None
        self.connected_at: datetime | None = None
        self.stopped_at: datetime | None = None

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._source_closed = False
        self._has_stopped = False

    async def start(self) -> None:
        """Start the background ingestion loop for this stream.

        Enforces single-use lifecycle: cannot restart after any terminal completion.
        Idempotent if already streaming or connecting.
        """
        if (
            self._has_stopped
            or self.state in (StreamState.STOPPED, StreamState.ERROR)
            or (self._loop_task is not None and self._loop_task.done())
        ):
            raise RuntimeError(
                f"StreamWorker '{self.config.stream_id}' has completed its lifecycle "
                "and cannot be restarted"
            )

        if self._loop_task is not None and not self._loop_task.done():
            return

        self._stop_event.clear()
        self.started_at = self._clock()
        self.state = StreamState.CONNECTING
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"stream-worker-{self.config.stream_id}",
        )

    async def _finalize_terminal(self) -> None:
        """Finalize worker state on any terminal exit path exactly once."""
        self._has_stopped = True
        self._stop_event.set()
        await self._close_source_safely()
        self.queue.close()
        if self.state != StreamState.ERROR:
            self.state = StreamState.STOPPED
        if self.stopped_at is None:
            self.stopped_at = self._clock()

    async def _sleep(self, delay: float) -> None:
        """Sleep with cancellation support via custom sleeper or asyncio.wait_for."""
        if self._stop_event.is_set():
            return

        if self._sleeper is not None:
            await self._sleeper(delay)
            return

        # Do NOT suppress CancelledError; let cancellation propagate cleanly
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)

    async def _close_source_safely(self) -> None:
        """Safely close underlying source exactly once without silently swallowing errors."""
        if self.source is not None and not self._source_closed:
            self._source_closed = True
            try:
                await self.source.close()
            except Exception as exc:
                self.last_error = f"close_error: {type(exc).__name__}"

    async def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self.state = StreamState.CONNECTING
                    if self.source is None:
                        raise SourceConnectionError(
                            f"No FrameSource configured for stream '{self.config.stream_id}'"
                        )

                    await self.source.connect()
                    self.connected_at = self._clock()
                    self.state = StreamState.STREAMING
                    self._consecutive_successful_frames = 0
                    # Reset per-connection session tracking and sampling baseline
                    self._current_session_id = None
                    self._last_sequence_number = None
                    self._last_sampled_timestamp = None

                    # Ingestion inner loop
                    while not self._stop_event.is_set():
                        frame = await self.source.read_frame()

                        if frame is None:
                            # Remote stream EOF or closed
                            if not self.config.is_live:
                                # Finite clip has ended cleanly; stop worker
                                self.state = StreamState.STOPPED
                                return

                            # Live camera disconnected unexpectedly
                            self.consecutive_failures += 1
                            self.connection_errors += 1
                            self.reconnect_attempts += 1
                            self._consecutive_successful_frames = 0
                            self.last_error = "connection_closed_eof"

                            max_fail = self.config.max_consecutive_failures
                            if max_fail is not None and self.consecutive_failures >= max_fail:
                                self.state = StreamState.ERROR
                                return

                            self.state = StreamState.BACKOFF
                            delay = self.backoff.compute_next_delay()
                            await self._sleep(delay)
                            break

                        # 1. Frame Identity Validation (stream_id & camera_external_id)
                        if frame.stream_id != self.config.stream_id:
                            self.integrity_errors += 1
                            self.last_error = "frame_integrity_error"
                            continue

                        if frame.camera_external_id != self.config.camera_external_id:
                            self.integrity_errors += 1
                            self.last_error = "frame_integrity_error"
                            continue

                        # 2. Session & Monotonic Sequence Semantics (MF05/MF06)
                        if self._current_session_id is None:
                            # First frame of connection establishes session and resets baseline
                            self._current_session_id = frame.session_id
                            self._last_sequence_number = frame.sequence_number
                            self._last_sampled_timestamp = None
                        else:
                            # Within the same connection, session_id must not change
                            if frame.session_id != self._current_session_id:
                                self.sequence_errors += 1
                                self.last_error = "session_sequence_error"
                                continue

                            # Within the same session, sequence_number must increase strictly
                            if (
                                self._last_sequence_number is not None
                                and frame.sequence_number <= self._last_sequence_number
                            ):
                                self.sequence_errors += 1
                                self.last_error = "session_sequence_error"
                                continue

                            self._last_sequence_number = frame.sequence_number

                        # 3. Stable Criteria: reset backoff after min_stable_frames
                        self._consecutive_successful_frames += 1
                        if self._consecutive_successful_frames >= self.config.min_stable_frames:
                            self.consecutive_failures = 0
                            self.backoff.reset()

                        # 4. Deterministic Sampling (target_fps limit)
                        if self.config.target_fps is not None:
                            min_interval = 1.0 / self.config.target_fps
                            if self._last_sampled_timestamp is not None:
                                elapsed = (
                                    frame.captured_at - self._last_sampled_timestamp
                                ).total_seconds()
                                if elapsed < (min_interval - 1e-6):
                                    self.sampled_out_frames += 1
                                    continue
                            self._last_sampled_timestamp = frame.captured_at

                        # 5. Enqueue frame into bounded drop-stale queue
                        self.queue.put(frame)
                        self.last_frame_timestamp = frame.captured_at

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self.last_error = classify_error_reason(exc)
                    self.consecutive_failures += 1
                    self._consecutive_successful_frames = 0
                    if isinstance(exc, SourceReadError):
                        self.read_errors += 1
                    elif isinstance(exc, FrameIntegrityError):
                        self.integrity_errors += 1
                    elif isinstance(exc, SessionSequenceError):
                        self.sequence_errors += 1
                    else:
                        self.connection_errors += 1
                    self.reconnect_attempts += 1

                    if (
                        self.config.max_consecutive_failures is not None
                        and self.consecutive_failures >= self.config.max_consecutive_failures
                    ):
                        self.state = StreamState.ERROR
                        break

                    self.state = StreamState.BACKOFF
                    delay = self.backoff.compute_next_delay()
                    await self._sleep(delay)
        finally:
            await self._finalize_terminal()

    async def stop(self) -> None:
        """Cooperatively stop the ingestion loop, unblock readers, and release resources."""
        self._has_stopped = True
        self._stop_event.set()

        # Unblock any downstream consumers waiting on queue.get()
        self.queue.close()

        # Actively close the source first to unblock any hung network/socket read_frame()
        await self._close_source_safely()

        if self._loop_task is not None:
            if not self._loop_task.done():
                self._loop_task.cancel()
                try:
                    await asyncio.wait_for(self._loop_task, timeout=2.0)
                except asyncio.CancelledError:
                    pass
                except TimeoutError:
                    self.last_error = "shutdown_timeout"
                except Exception as exc:
                    self.last_error = f"shutdown_error: {type(exc).__name__}"
            self._loop_task = None

        if self.state != StreamState.ERROR:
            self.state = StreamState.STOPPED
        if self.stopped_at is None:
            self.stopped_at = self._clock()

    async def get_frame(self) -> FrameEnvelope:
        """Asynchronously retrieve the next frame envelope from the bounded queue."""
        return await self.queue.get()

    def snapshot(self) -> StreamStatus:
        """Return an immutable status snapshot of this stream."""
        metrics = StreamMetrics(
            frames_enqueued=self.queue.enqueued_count,
            frames_dequeued=self.queue.dequeued_count,
            frames_dropped=self.queue.dropped_count,
            sampled_out_frames=self.sampled_out_frames,
            integrity_errors=self.integrity_errors,
            sequence_errors=self.sequence_errors,
            connection_errors=self.connection_errors,
            read_errors=self.read_errors,
            reconnect_attempts=self.reconnect_attempts,
            consecutive_failures=self.consecutive_failures,
            last_frame_timestamp=self.last_frame_timestamp,
            last_error=self.last_error,
            started_at=self.started_at,
            connected_at=self.connected_at,
            stopped_at=self.stopped_at,
        )
        return StreamStatus(
            stream_id=self.config.stream_id,
            camera_external_id=self.config.camera_external_id,
            sanitized_url=self.config.sanitized_source_url,
            state=self.state,
            metrics=metrics,
            queue_size=self.queue.qsize(),
            queue_capacity=self.queue.maxsize,
        )


class CameraIngestionWorker:
    """Manages multiple camera stream workers with fault isolation and health snapshots."""

    def __init__(self) -> None:
        self._streams: dict[str, StreamWorker] = {}
        self._status: Literal["idle", "running", "stopping", "stopped"] = "idle"

    def add_stream(
        self,
        config: StreamConfig,
        source: FrameSource | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> StreamWorker:
        """Register a new camera stream.

        Immutable configuration policy: streams must be registered in 'idle' state
        before the manager is started.

        Raises:
            RuntimeError: If manager is already running, stopping, or stopped.
            ValueError: If stream_id is already registered.
        """
        if self._status != "idle":
            raise RuntimeError(
                f"Cannot add stream '{config.stream_id}' while manager status is '{self._status}'. "
                "All streams must be registered before start()."
            )

        if config.stream_id in self._streams:
            raise ValueError(f"Stream with id '{config.stream_id}' already registered")

        worker = StreamWorker(config=config, source=source, sleeper=sleeper, clock=clock)
        self._streams[config.stream_id] = worker
        return worker

    def get_stream(self, stream_id: str) -> StreamWorker | None:
        """Lookup a registered stream worker."""
        return self._streams.get(stream_id)

    async def remove_stream(self, stream_id: str) -> None:
        """Stop and remove a stream worker."""
        worker = self._streams.pop(stream_id, None)
        if worker is not None:
            await worker.stop()

    async def start(self) -> None:
        """Start all registered stream workers.

        Can only be started from 'idle'. Idempotent if already running.
        Raises RuntimeError if manager has been stopped, is stopping, or is in an invalid state.
        """
        if self._status == "running":
            return

        if self._status == "stopped":
            raise RuntimeError("CameraIngestionWorker has been stopped and cannot be restarted")

        if self._status != "idle":
            raise RuntimeError(
                f"CameraIngestionWorker cannot be started when status is '{self._status}'"
            )

        try:
            await asyncio.gather(*(worker.start() for worker in self._streams.values()))
            self._status = "running"
        except Exception:
            self._status = "stopped"
            with contextlib.suppress(Exception):
                await asyncio.gather(*(worker.stop() for worker in self._streams.values()))
            raise

    async def stop(self) -> None:
        """Gracefully stop all stream workers and release resources."""
        if self._status == "stopped":
            return
        self._status = "stopping"
        try:
            await asyncio.gather(*(worker.stop() for worker in self._streams.values()))
        finally:
            self._status = "stopped"

    async def cancel(self) -> None:
        """Immediately cancel all stream workers."""
        await self.stop()

    async def release(self) -> None:
        """Release all stream queues and worker references."""
        await self.stop()
        for worker in self._streams.values():
            worker.queue.clear()
        self._streams.clear()

    def snapshot(self) -> WorkerStatus:
        """Generate an observable snapshot across all streams.

        Guarantees that inference_ready is always False in this camera ingestion foundation.
        """
        streams_snapshot = {sid: w.snapshot() for sid, w in self._streams.items()}
        active_count = sum(1 for s in streams_snapshot.values() if s.state == StreamState.STREAMING)

        return WorkerStatus(
            status=self._status,
            inference_ready=False,
            stream_count=len(self._streams),
            active_stream_count=active_count,
            streams=streams_snapshot,
        )
