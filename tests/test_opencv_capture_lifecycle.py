"""Threaded fake-capture lifecycle cancellation regressions; no native device use."""

import asyncio
import importlib
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Literal

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import CAP_PROP_FPS, OpenCvFrameSource


class GatedLifecycleCapture:
    def __init__(
        self,
        *,
        phase: Literal["open", "fps", "release"],
        on_enter: Callable[[], None],
        gate: threading.Event,
        worker_done: threading.Event,
    ) -> None:
        self.phase = phase
        self._on_enter = on_enter
        self._gate = gate
        self._worker_done = worker_done
        self._lock = threading.Lock()
        self.opened = True
        self.active_native_calls = 0
        self.busy = False
        self.open_calls = 0
        self.fps_calls = 0
        self.release_calls = 0
        self.release_completed = 0
        self.overlap_detected = False

    def _enter_native(self, *, is_release: bool = False) -> None:
        with self._lock:
            if self.active_native_calls > 0:
                self.overlap_detected = True
            if is_release:
                self.release_calls += 1
            self.active_native_calls += 1
            self.busy = True
        self._on_enter()

    def _exit_native(self, *, is_release: bool = False) -> None:
        with self._lock:
            if is_release:
                self.release_completed += 1
                self.opened = False
            self.active_native_calls -= 1
            self.busy = self.active_native_calls > 0
        self._worker_done.set()

    def block_in_open(self) -> "GatedLifecycleCapture":
        self.open_calls += 1
        if self.phase == "open":
            self._enter_native()
            try:
                assert self._gate.wait(timeout=2.0)
            finally:
                self._exit_native()
        return self

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop_id: int) -> float:
        if prop_id == CAP_PROP_FPS:
            self.fps_calls += 1
            if self.phase == "fps":
                self._enter_native()
                try:
                    assert self._gate.wait(timeout=2.0)
                    return 30.0
                finally:
                    self._exit_native()
            return 30.0
        return 0.0

    def release(self) -> None:
        if self.phase == "release":
            self._enter_native(is_release=True)
            try:
                assert self._gate.wait(timeout=2.0)
            finally:
                self._exit_native(is_release=True)
            return

        with self._lock:
            if self.active_native_calls > 0 or self.busy:
                self.overlap_detected = True
            self.release_calls += 1
            self.release_completed += 1
            self.opened = False


class FakeCv2:
    def __init__(self, capture: GatedLifecycleCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _source: int | str) -> GatedLifecycleCapture:
        return self.capture.block_in_open()


def install_fake_cv2(monkeypatch: pytest.MonkeyPatch, cv2: FakeCv2) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "cv2":
            return cv2
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def make_config(stream_id: str) -> StreamConfig:
    return StreamConfig(
        stream_id=stream_id,
        camera_external_id=f"cam-{stream_id}",
        source_url="rtsp://camera.local/live",
        is_live=True,
    )


async def spin(ticks: int = 10) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def test_cancel_connect_twice_during_blocked_open_releases_capture_once_after_open_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedLifecycleCapture(
            phase="open",
            on_enter=lambda: loop.call_soon_threadsafe(entered.set),
            gate=gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, FakeCv2(capture))
        source = OpenCvFrameSource(make_config("cancel-open"))

        connect_task: asyncio.Task[None] | None = None
        try:
            connect_task = asyncio.create_task(source.connect())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 0

            connect_task.cancel()
            await spin()
            assert capture.busy is True
            assert not connect_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            connect_task.cancel()
            await spin()
            assert capture.busy is True
            assert not connect_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(connect_task, timeout=1.0)

            assert capture.busy is False
            assert capture.release_calls == 1
            assert capture.release_completed == 1
            assert capture.opened is False
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if connect_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(connect_task, timeout=1.0)
            with suppress(BaseException):
                await asyncio.wait_for(source.close(), timeout=1.0)

    asyncio.run(scenario())


def test_cancel_connect_twice_during_fps_probe_waits_for_fps_drain_then_releases_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedLifecycleCapture(
            phase="fps",
            on_enter=lambda: loop.call_soon_threadsafe(entered.set),
            gate=gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, FakeCv2(capture))
        source = OpenCvFrameSource(make_config("cancel-fps"))

        connect_task: asyncio.Task[None] | None = None
        try:
            connect_task = asyncio.create_task(source.connect())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 0

            connect_task.cancel()
            await spin()
            assert capture.busy is True
            assert not connect_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            connect_task.cancel()
            await spin()
            assert capture.busy is True
            assert not connect_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(connect_task, timeout=1.0)

            assert capture.busy is False
            assert capture.release_calls == 1
            assert capture.release_completed == 1
            assert capture.opened is False
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if connect_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(connect_task, timeout=1.0)
            with suppress(BaseException):
                await asyncio.wait_for(source.close(), timeout=1.0)

    asyncio.run(scenario())


def test_cancel_close_twice_during_blocked_release_waits_and_deduplicates_second_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedLifecycleCapture(
            phase="release",
            on_enter=lambda: loop.call_soon_threadsafe(entered.set),
            gate=gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, FakeCv2(capture))
        source = OpenCvFrameSource(make_config("cancel-release"))
        await source.connect()
        assert source.is_connected is True

        first_close: asyncio.Task[None] | None = None
        second_close: asyncio.Task[None] | None = None
        try:
            first_close = asyncio.create_task(source.close())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 1
            assert capture.release_completed == 0

            second_close = asyncio.create_task(source.close())
            await spin()
            assert capture.busy is True
            assert not first_close.done()
            assert not second_close.done()
            assert capture.release_calls == 1
            assert not capture.overlap_detected

            first_close.cancel()
            await spin()
            assert capture.busy is True
            assert not first_close.done()
            assert not second_close.done()
            assert capture.release_calls == 1
            assert not capture.overlap_detected

            first_close.cancel()
            await spin()
            assert capture.busy is True
            assert not first_close.done()
            assert not second_close.done()
            assert capture.release_calls == 1
            assert not capture.overlap_detected

            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first_close, timeout=1.0)
            await asyncio.wait_for(second_close, timeout=1.0)

            assert capture.busy is False
            assert capture.release_calls == 1
            assert capture.release_completed == 1
            assert capture.opened is False
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if first_close is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(first_close, timeout=1.0)
            if second_close is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(second_close, timeout=1.0)
            with suppress(BaseException):
                await asyncio.wait_for(source.close(), timeout=1.0)

    asyncio.run(scenario())
