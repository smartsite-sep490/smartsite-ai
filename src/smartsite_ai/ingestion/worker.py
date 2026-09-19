"""Camera ingestion worker and per-stream worker lifecycle."""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Literal

from smartsite_ai.ingestion.backoff import ExponentialBackoff
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import BoundedFrameQueue
from smartsite_ai.ingestion.source import (
    FrameSource,
    SourceConnectionError,
    SourceReadError,
)
from smartsite_ai.ingestion.status import (
    StreamMetrics,
    StreamState,
    StreamStatus,
    WorkerStatus,
)


class StreamWorker:
    """Manages lifecycle, reconnection, backoff, and ingestion loop for a camera stream."""

    def __init__(self, config: StreamConfig, source: FrameSource | None = None) -> None:
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

        self.connection_errors = 0
        self.read_errors = 0
        self.reconnect_attempts = 0
        self.consecutive_failures = 0
        self.last_frame_timestamp: datetime | None = None
        self.last_error: str | None = None
        self.started_at: datetime | None = None
        self.connected_at: datetime | None = None
        self.stopped_at: datetime | None = None

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background ingestion loop for this stream."""
        if self._loop_task is not None and not self._loop_task.done():
            return
        self._stop_event.clear()
        self.started_at = datetime.now(UTC)
        self.state = StreamState.CONNECTING
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"stream-worker-{self.config.stream_id}",
        )

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.state = StreamState.CONNECTING
                if self.source is None:
                    raise SourceConnectionError(
                        f"No FrameSource configured for stream '{self.config.stream_id}'"
                    )

                await self.source.connect()
                self.connected_at = datetime.now(UTC)
                self.state = StreamState.STREAMING
                self.consecutive_failures = 0
                self.backoff.reset()

                # Read frames in loop until disconnect or cancellation
                while not self._stop_event.is_set():
                    frame = await self.source.read_frame()
                    if frame is None:
                        # Stream reached EOF or remote closed cleanly; back off before reconnecting
                        self.state = StreamState.BACKOFF
                        delay = self.backoff.compute_next_delay()
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                        break

                    self.queue.put(frame)
                    self.last_frame_timestamp = frame.captured_at

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error = str(exc)
                self.consecutive_failures += 1
                if isinstance(exc, SourceReadError):
                    self.read_errors += 1
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
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            finally:
                if self.source is not None and self.source.is_connected:
                    with contextlib.suppress(Exception):
                        await self.source.close()

        if self.state != StreamState.ERROR:
            self.state = StreamState.STOPPED
        self.stopped_at = datetime.now(UTC)

    async def stop(self) -> None:
        """Cooperatively stop the ingestion loop and release stream resources."""
        self._stop_event.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._loop_task
            self._loop_task = None

        if self.source is not None:
            with contextlib.suppress(Exception):
                await self.source.close()

        if self.state != StreamState.ERROR:
            self.state = StreamState.STOPPED

    async def get_frame(self) -> FrameEnvelope:
        """Asynchronously retrieve the next frame envelope from the bounded queue."""
        return await self.queue.get()

    def snapshot(self) -> StreamStatus:
        """Return an immutable status snapshot of this stream."""
        metrics = StreamMetrics(
            frames_enqueued=self.queue.enqueued_count,
            frames_dequeued=self.queue.dequeued_count,
            frames_dropped=self.queue.dropped_count,
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
    ) -> StreamWorker:
        """Register a new camera stream."""
        worker = StreamWorker(config=config, source=source)
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
        """Start all registered stream workers."""
        self._status = "running"
        for worker in self._streams.values():
            await worker.start()

    async def stop(self) -> None:
        """Gracefully stop all stream workers and release resources."""
        self._status = "stopping"
        for worker in self._streams.values():
            await worker.stop()
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
