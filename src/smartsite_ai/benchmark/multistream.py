"""Deterministic scheduling and measurement for shared-detector video replay."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import UUID, uuid4

from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.envelope import FrameEnvelope


class BenchmarkError(RuntimeError):
    """The benchmark could not produce a complete, trustworthy scenario."""


@dataclass(frozen=True, slots=True)
class ReplaySourceFacts:
    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1 or self.frame_count < 1:
            raise ValueError("replay source dimensions and frame count must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("replay source FPS must be finite and positive")


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    payload: bytes
    decode_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("decoded frame payload must be non-empty bytes")
        if not math.isfinite(self.decode_seconds) or self.decode_seconds < 0:
            raise ValueError("decode latency must be finite and non-negative")


class ReplaySourceProtocol(Protocol):
    @property
    def facts(self) -> ReplaySourceFacts: ...

    def read(self) -> DecodedFrame: ...

    def skip(self, frame_count: int) -> None: ...

    def close(self) -> None: ...


class InferenceSynchronizerProtocol(Protocol):
    def synchronize(self) -> None: ...


class ResourceMonitorProtocol(Protocol):
    def start(self) -> None: ...

    def sample(self) -> None: ...

    def finish(self) -> Mapping[str, object]: ...


ReplaySourceFactory = Callable[[Path], ReplaySourceProtocol]
Clock = Callable[[], float]
SessionFactory = Callable[[], UUID]
UtcNowFactory = Callable[[], datetime]


@dataclass(slots=True)
class _LatencySamples:
    decode: list[float] = field(default_factory=list)
    queue: list[float] = field(default_factory=list)
    inference: list[float] = field(default_factory=list)
    pipeline: list[float] = field(default_factory=list)
    end_to_end: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _StreamAccumulator:
    stream_id: str
    input_path: Path
    source_fps: float
    effective_target_fps: float
    processed_frames: int = 0
    dropped_frames: int = 0
    late_completed_frames: int = 0
    latencies: _LatencySamples = field(default_factory=_LatencySamples)


@dataclass(frozen=True, slots=True)
class _InferenceTiming:
    queue_seconds: float
    inference_seconds: float
    batch: DetectionBatch


class _SharedInferenceLane:
    """Serialize access to one detector/runner and measure contention separately."""

    def __init__(
        self,
        detector: DetectorProtocol,
        synchronizer: InferenceSynchronizerProtocol,
        clock: Clock,
    ) -> None:
        self._detector = detector
        self._synchronizer = synchronizer
        self._clock = clock
        self._lock = asyncio.Lock()

    async def detect(self, frame: FrameEnvelope) -> _InferenceTiming:
        queued_at = self._clock()
        async with self._lock:
            acquired_at = self._clock()
            self._synchronizer.synchronize()
            inference_started = self._clock()
            batch = await self._detector.detect(frame)
            self._synchronizer.synchronize()
            inference_finished = self._clock()
        if not isinstance(batch, DetectionBatch):
            raise BenchmarkError("detector returned an invalid detection batch")
        return _InferenceTiming(
            queue_seconds=max(acquired_at - queued_at, 0.0),
            inference_seconds=max(inference_finished - inference_started, 0.0),
            batch=batch,
        )


def _percentiles_seconds(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50Ms": None, "p95Ms": None, "p99Ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = ordered[lower]
        else:
            value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        return round(value * 1_000.0, 6)

    return {"p50Ms": percentile(0.50), "p95Ms": percentile(0.95), "p99Ms": percentile(0.99)}


def _latency_report(samples: _LatencySamples) -> dict[str, object]:
    return {
        "sampleCount": len(samples.inference),
        "percentileEstimator": "linear-interpolation-at-rank-(n-1)*q",
        "decodeAndPack": _percentiles_seconds(samples.decode),
        "inferenceQueueWait": _percentiles_seconds(samples.queue),
        "detector": _percentiles_seconds(samples.inference),
        "pipeline": _percentiles_seconds(samples.pipeline),
        "endToEnd": _percentiles_seconds(samples.end_to_end),
    }


async def _sleep_until(deadline: float, clock: Clock) -> None:
    remaining = deadline - clock()
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _run_stream_phase(
    *,
    source: ReplaySourceProtocol,
    accumulator: _StreamAccumulator,
    lane: _SharedInferenceLane,
    phase_start: float,
    duration_seconds: float,
    collect: bool,
    clock: Clock,
    session_id: UUID,
    captured_at_start: datetime,
    starting_sequence: int,
) -> int:
    interval = 1.0 / accumulator.effective_target_fps
    phase_end = phase_start + duration_seconds
    total_slots = max(0, math.ceil(duration_seconds / interval - 1e-12))
    slot = 0
    sequence_number = starting_sequence
    await _sleep_until(phase_start, clock)

    while slot < total_slots:
        next_due = phase_start + slot * interval
        await _sleep_until(next_due, clock)
        now = clock()
        if now >= phase_end:
            break
        missed = min(max(0, int((now - next_due) // interval)), total_slots - slot)
        if missed:
            if collect:
                accumulator.dropped_frames += missed
            try:
                await asyncio.to_thread(source.skip, missed)
            except Exception as error:
                raise BenchmarkError(
                    f"could not advance {accumulator.stream_id} past dropped frames"
                ) from error
            sequence_number += missed
            slot += missed
            if slot >= total_slots:
                break
            next_due = phase_start + slot * interval

        pipeline_started = clock()
        try:
            decoded = await asyncio.to_thread(source.read)
        except Exception as error:
            raise BenchmarkError(f"could not decode {accumulator.stream_id}") from error
        facts = source.facts
        if len(decoded.payload) != facts.width * facts.height * 3:
            raise BenchmarkError(f"{accumulator.stream_id} returned an invalid BGR24 payload")
        frame = FrameEnvelope(
            stream_id=accumulator.stream_id,
            session_id=session_id,
            camera_external_id=accumulator.stream_id,
            captured_at=captured_at_start + timedelta(seconds=max(next_due - phase_start, 0.0)),
            width=facts.width,
            height=facts.height,
            sequence_number=sequence_number,
            payload=decoded.payload,
        )
        try:
            timing = await lane.detect(frame)
        except Exception as error:
            if isinstance(error, BenchmarkError):
                raise
            raise BenchmarkError(f"detector failed for {accumulator.stream_id}") from error
        completed_at = clock()
        if (
            timing.batch.stream_id != frame.stream_id
            or timing.batch.sequence_number != sequence_number
        ):
            raise BenchmarkError("detector batch identity does not match its input frame")

        if collect:
            accumulator.latencies.decode.append(decoded.decode_seconds)
            accumulator.latencies.queue.append(timing.queue_seconds)
            accumulator.latencies.inference.append(timing.inference_seconds)
            accumulator.latencies.pipeline.append(max(completed_at - pipeline_started, 0.0))
            accumulator.latencies.end_to_end.append(max(completed_at - next_due, 0.0))
            if completed_at <= phase_end:
                accumulator.processed_frames += 1
            else:
                # The work really ran, but counting a completion after the fixed window would
                # overstate real-time throughput. Treat its scheduled slot as missed instead.
                accumulator.dropped_frames += 1
                accumulator.late_completed_frames += 1
        sequence_number += 1
        slot += 1

    if collect:
        accumulator.dropped_frames += total_slots - slot
    return sequence_number


async def _measure_resources(
    *,
    monitor: ResourceMonitorProtocol,
    phase_start: float,
    duration_seconds: float,
    interval_seconds: float,
    clock: Clock,
) -> dict[str, object]:
    await _sleep_until(phase_start, clock)
    phase_end = phase_start + duration_seconds
    while clock() < phase_end:
        monitor.sample()
        await asyncio.sleep(min(interval_seconds, max(phase_end - clock(), 0.0)))
    monitor.sample()
    return dict(monitor.finish())


async def run_multistream_scenario(
    *,
    detector: DetectorProtocol,
    input_paths: Sequence[Path],
    source_factory: ReplaySourceFactory,
    synchronizer: InferenceSynchronizerProtocol,
    resource_monitor: ResourceMonitorProtocol,
    warmup_seconds: float,
    measurement_seconds: float,
    target_fps: float | None = None,
    resource_sample_interval_seconds: float = 0.1,
    clock: Clock = perf_counter,
    session_factory: SessionFactory = uuid4,
    utc_now_factory: UtcNowFactory = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Run one fixed-duration scenario through one serialized shared detector."""

    if not 1 <= len(input_paths) <= 3:
        raise BenchmarkError("scenario must contain between one and three streams")
    for label, value, allow_zero in (
        ("warm-up duration", warmup_seconds, True),
        ("measurement duration", measurement_seconds, False),
        ("resource sample interval", resource_sample_interval_seconds, False),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value <= 0):
            raise BenchmarkError(
                f"{label} must be finite and {'non-negative' if allow_zero else 'positive'}"
            )
    if target_fps is not None and (not math.isfinite(target_fps) or target_fps <= 0):
        raise BenchmarkError("target FPS must be finite and positive")

    sources: list[ReplaySourceProtocol] = []
    primary_error: BaseException | None = None
    try:
        for path in input_paths:
            sources.append(source_factory(path))
        accumulators = [
            _StreamAccumulator(
                stream_id=f"benchmark-stream-{index + 1}",
                input_path=path,
                source_fps=source.facts.fps,
                effective_target_fps=target_fps or source.facts.fps,
            )
            for index, (path, source) in enumerate(zip(input_paths, sources, strict=True))
        ]
        lane = _SharedInferenceLane(detector, synchronizer, clock)
        sessions = [session_factory() for _ in sources]
        if any(not isinstance(session, UUID) for session in sessions):
            raise BenchmarkError("session factory must return UUID values")
        sequence_numbers = [0 for _ in sources]
        captured_at_start = utc_now_factory()
        if captured_at_start.tzinfo is None or captured_at_start.utcoffset() is None:
            raise BenchmarkError("UTC clock must return a timezone-aware datetime")

        if warmup_seconds > 0:
            warmup_start = clock() + 0.01
            sequence_numbers = list(
                await asyncio.gather(
                    *(
                        _run_stream_phase(
                            source=source,
                            accumulator=accumulator,
                            lane=lane,
                            phase_start=warmup_start,
                            duration_seconds=warmup_seconds,
                            collect=False,
                            clock=clock,
                            session_id=session,
                            captured_at_start=captured_at_start,
                            starting_sequence=sequence,
                        )
                        for source, accumulator, session, sequence in zip(
                            sources, accumulators, sessions, sequence_numbers, strict=True
                        )
                    )
                )
            )

        resource_monitor.start()
        measurement_start = clock()
        resource_task = asyncio.create_task(
            _measure_resources(
                monitor=resource_monitor,
                phase_start=measurement_start,
                duration_seconds=measurement_seconds,
                interval_seconds=resource_sample_interval_seconds,
                clock=clock,
            )
        )
        try:
            await asyncio.gather(
                *(
                    _run_stream_phase(
                        source=source,
                        accumulator=accumulator,
                        lane=lane,
                        phase_start=measurement_start,
                        duration_seconds=measurement_seconds,
                        collect=True,
                        clock=clock,
                        session_id=session,
                        captured_at_start=captured_at_start,
                        starting_sequence=sequence,
                    )
                    for source, accumulator, session, sequence in zip(
                        sources, accumulators, sessions, sequence_numbers, strict=True
                    )
                )
            )
            resources = await resource_task
            measurement_finished = clock()
        finally:
            if not resource_task.done():
                resource_task.cancel()
                with suppress(asyncio.CancelledError):
                    await resource_task
        streams = []
        for accumulator in accumulators:
            total = accumulator.processed_frames + accumulator.dropped_frames
            streams.append(
                {
                    "streamId": accumulator.stream_id,
                    "inputPath": str(accumulator.input_path),
                    "sourceFps": accumulator.source_fps,
                    "targetFps": accumulator.effective_target_fps,
                    "scheduledFrames": total,
                    "processedFrames": accumulator.processed_frames,
                    "droppedFrames": accumulator.dropped_frames,
                    "lateCompletedFrames": accumulator.late_completed_frames,
                    "completedInferenceFrames": (
                        accumulator.processed_frames + accumulator.late_completed_frames
                    ),
                    "effectiveFps": round(accumulator.processed_frames / measurement_seconds, 6),
                    "latency": _latency_report(accumulator.latencies),
                }
            )
        return {
            "streamCount": len(streams),
            "warmupSeconds": warmup_seconds,
            "measurementSeconds": measurement_seconds,
            "executionElapsedSeconds": round(max(measurement_finished - measurement_start, 0.0), 6),
            "streams": streams,
            "aggregate": {
                "processedFrames": sum(item["processedFrames"] for item in streams),
                "droppedFrames": sum(item["droppedFrames"] for item in streams),
                "lateCompletedFrames": sum(item["lateCompletedFrames"] for item in streams),
                "completedInferenceFrames": sum(
                    item["completedInferenceFrames"] for item in streams
                ),
                "effectiveFps": round(
                    sum(item["processedFrames"] for item in streams) / measurement_seconds,
                    6,
                ),
            },
            "resources": resources,
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_errors: list[Exception] = []
        for source in sources:
            try:
                source.close()
            except Exception as error:
                close_errors.append(error)
        if close_errors and primary_error is None:
            raise BenchmarkError("could not close one or more replay sources") from close_errors[0]


__all__ = [
    "BenchmarkError",
    "DecodedFrame",
    "InferenceSynchronizerProtocol",
    "ReplaySourceFacts",
    "ReplaySourceProtocol",
    "ResourceMonitorProtocol",
    "run_multistream_scenario",
]
