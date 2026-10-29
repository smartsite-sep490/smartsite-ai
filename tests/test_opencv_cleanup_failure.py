"""Native cleanup failure must remain visible through cancellation and later calls."""

import asyncio
import importlib
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Literal

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.source import SourceConnectionError
from smartsite_ai.ingestion.status import StreamState
from smartsite_ai.ingestion.worker import StreamWorker


class GatedFailingReleaseCapture:
    def __init__(
        self,
        *,
        block_open: bool,
        on_open_enter: Callable[[], None] | None = None,
        open_gate: threading.Event | None = None,
        on_release_enter: Callable[[], None],
        release_gate: threading.Event,
        worker_done: threading.Event,
    ) -> None:
        self._block_open = block_open
        self._on_open_enter = on_open_enter
        self._open_gate = open_gate
        self._on_release_enter = on_release_enter
        self._release_gate = release_gate
        self._worker_done = worker_done
        self._lock = threading.Lock()
        self.active_native_calls = 0
        self.busy = False
        self.overlap_detected = False
        self.open_calls = 0
        self.release_calls = 0
        self.release_completed = 0

    def _enter(self, *, is_release: bool = False) -> None:
        with self._lock:
            if self.active_native_calls > 0:
                self.overlap_detected = True
            if is_release:
                self.release_calls += 1
            self.active_native_calls += 1
            self.busy = True

    def _exit(self, *, is_release: bool = False) -> None:
        with self._lock:
            if is_release:
                self.release_completed += 1
            self.active_native_calls -= 1
            self.busy = self.active_native_calls > 0
            if is_release:
                self._worker_done.set()

    def open_handle(self) -> "GatedFailingReleaseCapture":
        self.open_calls += 1
        if self._block_open:
            assert self._open_gate is not None
            assert self._on_open_enter is not None
            self._enter()
            self._on_open_enter()
            try:
                assert self._open_gate.wait(timeout=2.0)
            finally:
                self._exit()
        return self

    def isOpened(self) -> bool:
        return True

    def get(self, _prop_id: int) -> float:
        return 30.0

    def release(self) -> None:
        self._enter(is_release=True)
        self._on_release_enter()
        try:
            assert self._release_gate.wait(timeout=2.0)
            raise RuntimeError("rtsp://user:secret-token@camera.local/native-release-failed")
        finally:
            self._exit(is_release=True)


class ValidationFailureCapture:
    def __init__(self, *, mode: Literal["return_false", "raise_error"]) -> None:
        self._mode = mode
        self.release_calls = 0

    def isOpened(self) -> bool:
        if self._mode == "raise_error":
            raise ValueError("rtsp://user:secret-token@camera.local/is-opened-failed")
        return False

    def release(self) -> None:
        self.release_calls += 1
        raise RuntimeError("rtsp://user:secret-token@camera.local/cleanup-release-failed")


def install_fake_cv2(
    monkeypatch: pytest.MonkeyPatch, factory: Callable[[int | str], object]
) -> None:
    real_import_module = importlib.import_module

    class _Cv2Module:
        @staticmethod
        def VideoCapture(source: int | str) -> object:
            return factory(source)

    cv2_module = _Cv2Module()

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "cv2":
            return cv2_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def make_config(stream_id: str) -> StreamConfig:
    return StreamConfig(
        stream_id=stream_id,
        camera_external_id=f"cam-{stream_id}",
        source_url="rtsp://user:secret-token@camera.local/live",
        is_live=True,
    )


async def spin(ticks: int = 10) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def test_cancel_close_twice_when_native_release_raises_surfaces_source_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        release_entered = asyncio.Event()
        release_gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedFailingReleaseCapture(
            block_open=False,
            on_release_enter=lambda: loop.call_soon_threadsafe(release_entered.set),
            release_gate=release_gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, lambda _src: capture.open_handle())
        source = OpenCvFrameSource(make_config("close-release-fail"))
        await source.connect()
        assert source.is_connected is True

        close_task: asyncio.Task[None] | None = None
        try:
            close_task = asyncio.create_task(source.close())
            await asyncio.wait_for(release_entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 1

            close_task.cancel()
            await spin()
            assert capture.busy is True
            assert not close_task.done()

            close_task.cancel()
            await spin()
            assert capture.busy is True
            assert not close_task.done()

            release_gate.set()
            with pytest.raises(SourceConnectionError) as exc_info:
                await asyncio.wait_for(close_task, timeout=1.0)

            message = str(exc_info.value)
            assert "RuntimeError" in message
            assert "secret-token" not in message
            assert capture.busy is False
            assert capture.release_calls == 1
            assert capture.release_completed == 1
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            release_gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if close_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(close_task, timeout=1.0)

    asyncio.run(scenario())


def test_late_open_discard_failure_survives_second_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        open_entered = asyncio.Event()
        release_entered = asyncio.Event()
        open_gate = threading.Event()
        release_gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedFailingReleaseCapture(
            block_open=True,
            on_open_enter=lambda: loop.call_soon_threadsafe(open_entered.set),
            open_gate=open_gate,
            on_release_enter=lambda: loop.call_soon_threadsafe(release_entered.set),
            release_gate=release_gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, lambda _src: capture.open_handle())
        source = OpenCvFrameSource(make_config("late-open-discard-fail"))

        connect_task: asyncio.Task[None] | None = None
        try:
            connect_task = asyncio.create_task(source.connect())
            await asyncio.wait_for(open_entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 0

            # First cancellation during blocked native VideoCapture open.
            connect_task.cancel()
            await spin()
            assert not connect_task.done()
            assert capture.release_calls == 0

            # Unblock open so discard_on_cancel begins native release.
            open_gate.set()
            await asyncio.wait_for(release_entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 1
            assert not connect_task.done()

            # Second cancellation while discard_on_cancel is blocked in native release.
            connect_task.cancel()
            await spin()
            assert capture.busy is True
            assert not connect_task.done()

            release_gate.set()
            with pytest.raises(SourceConnectionError) as exc_info:
                await asyncio.wait_for(connect_task, timeout=1.0)

            message = str(exc_info.value)
            assert "RuntimeError" in message
            assert "secret-token" not in message
            assert capture.busy is False
            assert capture.release_calls == 1
            assert capture.release_completed == 1
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            open_gate.set()
            release_gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if connect_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(connect_task, timeout=1.0)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["return_false", "raise_error"])
def test_open_validation_failure_with_failing_release_surfaces_safe_source_connection_error(
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["return_false", "raise_error"],
) -> None:
    async def scenario() -> None:
        capture = ValidationFailureCapture(mode=mode)
        install_fake_cv2(monkeypatch, lambda _src: capture)
        source = OpenCvFrameSource(make_config(f"open-validate-{mode}"))

        with pytest.raises(SourceConnectionError) as exc_info:
            await source.connect()

        message = str(exc_info.value)
        assert "secret-token" not in message
        assert capture.release_calls == 1
        assert source.is_connected is False

    asyncio.run(scenario())


def test_release_failure_blocks_later_close_and_connect_without_native_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        gate = threading.Event()
        gate.set()
        capture = GatedFailingReleaseCapture(
            block_open=False,
            on_release_enter=lambda: None,
            release_gate=gate,
            worker_done=threading.Event(),
        )
        install_fake_cv2(monkeypatch, lambda _src: capture.open_handle())
        source = OpenCvFrameSource(make_config("release-failure-latch"))
        await source.connect()
        with pytest.raises(SourceConnectionError):
            await source.close()
        for operation in (source.close, source.connect):
            with pytest.raises(SourceConnectionError):
                await operation()
        assert capture.open_calls == 1
        assert capture.release_calls == 1
        assert not source.is_connected

    asyncio.run(scenario())


def test_worker_reports_error_after_failed_open_cleanup_without_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        capture = ValidationFailureCapture(mode="return_false")
        opens = []

        def factory(source: int | str) -> object:
            opens.append(source)
            return capture

        install_fake_cv2(monkeypatch, factory)
        config = make_config("worker-cleanup-error")
        worker = StreamWorker(config, source=OpenCvFrameSource(config))
        await worker.start()
        try:
            assert worker._loop_task is not None
            await asyncio.wait_for(asyncio.shield(worker._loop_task), timeout=1.0)
            assert worker.snapshot().state == StreamState.ERROR
            assert worker.last_error is not None
            assert worker.last_error.startswith("close_error:")
            assert len(opens) == 1
            assert capture.release_calls == 1
        finally:
            await asyncio.wait_for(worker.stop(), timeout=1.0)
        assert worker.snapshot().state == StreamState.ERROR

    asyncio.run(scenario())
