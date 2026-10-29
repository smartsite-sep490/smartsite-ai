"""Run isolated camera sessions and aggregate their terminal states."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from smartsite_ai.inference.protocol import DetectorProtocol
from smartsite_ai.ingestion.status import StreamMetrics
from smartsite_ai.integrations.outbox import OutboxCounts
from smartsite_ai.processing_worker import ProcessingWorkerResult, frame_flow_payload

_LOGGER = logging.getLogger(__name__)


class _Worker(Protocol):
    async def run(self) -> ProcessingWorkerResult: ...


class _Poller(Protocol):
    async def run(self, stop_event: asyncio.Event) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraSession:
    stream_id: str
    camera_external_id: str
    worker: _Worker
    poller: _Poller


@dataclass(frozen=True, slots=True)
class CameraOutcome:
    stream_id: str
    camera_external_id: str
    status: str
    frames_processed: int = 0
    events_enqueued: int = 0
    outbox: OutboxCounts | None = None
    error_class: str | None = None
    stream_metrics: StreamMetrics | None = None


@dataclass(slots=True)
class RuntimeReport:
    cameras: tuple[CameraOutcome, ...]
    model_loaded: bool
    model_closed: bool


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    preflight: Callable[[], Awaitable[tuple[CameraOutcome, ...] | None]]
    open_model: Callable[[], Awaitable[DetectorProtocol]]
    close_model: Callable[[], Awaitable[None]]
    build_sessions: Callable[[DetectorProtocol], Sequence[CameraSession]]


def not_started(stream_id: str, camera_external_id: str) -> CameraOutcome:
    return CameraOutcome(
        stream_id=stream_id,
        camera_external_id=camera_external_id,
        status="not_started",
    )


def failed(stream_id: str, camera_external_id: str, error_class: str) -> CameraOutcome:
    return CameraOutcome(
        stream_id=stream_id,
        camera_external_id=camera_external_id,
        status="failed",
        error_class=error_class,
    )


def startup_outcomes(
    cameras: Sequence[tuple[str, str]],
    failures: Sequence[tuple[str, str]],
) -> tuple[CameraOutcome, ...]:
    """Mark every camera not started when any preflight check fails."""

    failure_by_stream = {stream_id: error_class for stream_id, error_class in failures}
    if not failure_by_stream:
        raise ValueError("startup outcomes require at least one failure")
    outcomes: list[CameraOutcome] = []
    for stream_id, camera_external_id in cameras:
        error_class = failure_by_stream.get(stream_id)
        if error_class is None:
            outcomes.append(not_started(stream_id, camera_external_id))
        else:
            outcomes.append(failed(stream_id, camera_external_id, error_class))
    return tuple(outcomes)


async def run_camera_session(session: CameraSession) -> ProcessingWorkerResult:
    """Run one worker and its poller. A failure here does not know about siblings."""

    stop_polling = asyncio.Event()
    worker_task = asyncio.create_task(session.worker.run(), name=f"worker-{session.stream_id}")
    poller_task = asyncio.create_task(
        session.poller.run(stop_polling),
        name=f"poller-{session.stream_id}",
    )
    try:
        done, _pending = await asyncio.wait(
            (worker_task, poller_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if poller_task in done:
            if not worker_task.done():
                worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
            poller_task.result()
            raise RuntimeError("configuration poller stopped unexpectedly")
        result = worker_task.result()
        stop_polling.set()
        await poller_task
        return result
    finally:
        stop_polling.set()
        for task in (worker_task, poller_task):
            if not task.done():
                task.cancel()
        async with _hold_cancellation():
            await asyncio.gather(worker_task, poller_task, return_exceptions=True)


async def supervise_sessions(sessions: Sequence[CameraSession]) -> tuple[CameraOutcome, ...]:
    """Run sessions together and keep a healthy camera alive after another fails."""

    if not 1 <= len(sessions) <= 3:
        raise ValueError("runtime requires 1 to 3 cameras")
    tasks = [
        asyncio.create_task(run_camera_session(session), name=f"camera-{session.stream_id}")
        for session in sessions
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        async with _hold_cancellation():
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return tuple(
        _outcome_from_result(session, result)
        for session, result in zip(sessions, results, strict=True)
    )


async def execute_runtime(plan: RuntimePlan) -> RuntimeReport:
    """Preflight every camera, then load the model once around the supervised run."""

    report = RuntimeReport(cameras=(), model_loaded=False, model_closed=False)
    try:
        startup_failure = await plan.preflight()
        if startup_failure is not None:
            report.cameras = startup_failure
            return report
        detector = await plan.open_model()
        report.model_loaded = True
        report.cameras = await supervise_sessions(plan.build_sessions(detector))
        return report
    finally:
        if report.model_loaded and not report.model_closed:
            async with _hold_cancellation():
                await plan.close_model()
                report.model_closed = True


def exit_code(report: RuntimeReport) -> int:
    """Match the one-camera codes: 0 clean, 2 durable review, 1 failed startup or run."""

    if not report.cameras:
        return 1
    if any(camera.status != "completed" for camera in report.cameras):
        return 1
    if any(
        camera.outbox is None or camera.outbox.pending != 0 or camera.outbox.terminal != 0
        for camera in report.cameras
    ):
        return 2
    return 0


def report_payload(report: RuntimeReport) -> dict[str, object]:
    cameras: list[dict[str, object]] = []
    for camera in report.cameras:
        payload: dict[str, object] = {
            "streamId": camera.stream_id,
            "cameraExternalId": camera.camera_external_id,
            "status": camera.status,
            "framesProcessed": camera.frames_processed,
            "eventsEnqueued": camera.events_enqueued,
        }
        if camera.outbox is not None:
            payload["outbox"] = {
                "pending": camera.outbox.pending,
                "delivered": camera.outbox.delivered,
                "terminal": camera.outbox.terminal,
            }
        if camera.error_class is not None:
            payload["errorClass"] = camera.error_class
        frame_flow = frame_flow_payload(camera.stream_metrics)
        if frame_flow is not None:
            payload["frameFlow"] = frame_flow
        cameras.append(payload)
    return {
        "modelLoaded": report.model_loaded,
        "modelClosed": report.model_closed,
        "cameras": cameras,
    }


@asynccontextmanager
async def _hold_cancellation() -> AsyncIterator[None]:
    """Finish cleanup awaits after this task has already been cancelled.

    Python 3.11 injects ``CancelledError`` at the next await while the task is
    cancelling. Suspend that count only for the cleanup await, then restore it.
    """

    task = asyncio.current_task()
    suspended = 0
    if task is not None:
        while task.cancelling():
            task.uncancel()
            suspended += 1
    try:
        yield
    finally:
        if task is not None:
            for _ in range(suspended):
                task.cancel()


def _outcome_from_result(session: CameraSession, result: object) -> CameraOutcome:
    if isinstance(result, ProcessingWorkerResult):
        return CameraOutcome(
            stream_id=session.stream_id,
            camera_external_id=session.camera_external_id,
            status="completed",
            frames_processed=result.frames_processed,
            events_enqueued=result.events_enqueued,
            outbox=result.outbox,
            stream_metrics=result.stream_metrics,
        )
    if isinstance(result, Exception):
        _LOGGER.warning(
            "camera session failed",
            extra={
                "stream_id": session.stream_id,
                "camera_external_id": session.camera_external_id,
                "error_class": type(result).__name__,
            },
        )
        return failed(session.stream_id, session.camera_external_id, type(result).__name__)
    raise RuntimeError("camera session returned an unexpected result")


__all__ = [
    "CameraOutcome",
    "CameraSession",
    "RuntimePlan",
    "RuntimeReport",
    "execute_runtime",
    "exit_code",
    "failed",
    "not_started",
    "report_payload",
    "run_camera_session",
    "startup_outcomes",
    "supervise_sessions",
]
