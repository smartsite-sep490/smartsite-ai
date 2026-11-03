"""Deterministic threaded cancellation regression tests for OpenCvFrameSource (AI-A04)."""

import asyncio
import importlib
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Literal

import pytest

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import CAP_PROP_POS_MSEC, OpenCvFrameSource


class FakeFrame:
    def __init__(self, data: bytes, *, shape: tuple[int, int, int]) -> None:
        self._data = data
        self.shape = shape
        self.dtype = "uint8"
        self.flags = {"C_CONTIGUOUS": True}

    def tobytes(self) -> bytes:
        return self._data


def make_bgr_frame(width: int = 2, height: int = 2, fill: int = 1) -> FakeFrame:
    return FakeFrame(bytes([fill]) * width * height * 3, shape=(height, width, 3))


class GatedThreadedCapture:
    def __init__(
        self,
        *,
        phase: Literal["read", "get_POS_MSEC"],
        on_enter: Callable[[], None],
        release_gate: threading.Event,
        worker_done: threading.Event,
        native_raises: bool = False,
    ) -> None:
        self.phase = phase
        self._on_enter = on_enter
        self._release_gate = release_gate
        self._worker_done = worker_done
        self.native_raises = native_raises
        self._lock = threading.Lock()
        self.opened = True
        self.active_native_calls = 0
        self.busy = False
        self.read_calls = 0
        self.pos_msec_calls = 0
        self.release_calls = 0
        self.overlap_detected = False

    def _enter_native(self) -> None:
        with self._lock:
            self.active_native_calls += 1
            self.busy = True
        self._on_enter()

    def _exit_native(self) -> None:
        with self._lock:
            self.active_native_calls -= 1
            self.busy = self.active_native_calls > 0
        self._worker_done.set()

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        self.read_calls += 1
        if self.phase == "read":
            self._enter_native()
            try:
                assert self._release_gate.wait(timeout=2.0)
                if self.native_raises:
                    raise RuntimeError("simulated native read error after cancel")
                return True, make_bgr_frame()
            finally:
                self._exit_native()
        return True, make_bgr_frame()

    def get(self, prop_id: int) -> float:
        if prop_id == CAP_PROP_POS_MSEC:
            self.pos_msec_calls += 1
            if self.phase == "get_POS_MSEC":
                self._enter_native()
                try:
                    assert self._release_gate.wait(timeout=2.0)
                    if self.native_raises:
                        raise RuntimeError("simulated native timestamp error after cancel")
                    return 100.0
                finally:
                    self._exit_native()
            return 100.0
        return 30.0

    def release(self) -> None:
        with self._lock:
            if self.active_native_calls > 0 or self.busy:
                self.overlap_detected = True
            self.release_calls += 1
            self.opened = False


class FakeCv2:
    def __init__(self, capture: GatedThreadedCapture) -> None:
        self.capture = capture

    def VideoCapture(self, _source: int | str) -> GatedThreadedCapture:
        return self.capture


def install_fake_cv2(monkeypatch: pytest.MonkeyPatch, cv2: FakeCv2) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "cv2":
            return cv2
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


async def spin(ticks: int = 10) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.parametrize("phase", ["read", "get_POS_MSEC"])
def test_cancelled_read_frame_waits_for_in_flight_native_call_before_close_release(
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["read", "get_POS_MSEC"],
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release_gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedThreadedCapture(
            phase=phase,
            on_enter=lambda: loop.call_soon_threadsafe(entered.set),
            release_gate=release_gate,
            worker_done=worker_done,
        )
        install_fake_cv2(monkeypatch, FakeCv2(capture))

        is_live = phase == "read"
        config = StreamConfig(
            stream_id=f"stream-{phase}",
            camera_external_id=f"cam-{phase}",
            source_url="rtsp://camera.local/live" if is_live else r"D:\data\clip.mp4",
            is_live=is_live,
        )
        source = OpenCvFrameSource(config)
        await source.connect()

        read_task: asyncio.Task[object] | None = None
        close_task: asyncio.Task[None] | None = None
        try:
            read_task = asyncio.create_task(source.read_frame())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert capture.busy is True
            assert capture.release_calls == 0

            close_task = asyncio.create_task(source.close())
            await spin()
            assert capture.busy is True
            assert not close_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            read_task.cancel()
            await spin()
            assert capture.busy is True
            assert not close_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            read_task.cancel()
            await spin()
            assert capture.busy is True
            assert not close_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            release_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read_task, timeout=1.0)
            await asyncio.wait_for(close_task, timeout=1.0)

            assert capture.busy is False
            assert capture.release_calls == 1
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            release_gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if read_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(read_task, timeout=1.0)
            if close_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(close_task, timeout=1.0)
            with suppress(BaseException):
                await asyncio.wait_for(source.close(), timeout=1.0)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["read", "get_POS_MSEC"])
def test_cancelled_read_frame_preserves_cancelled_error_when_native_call_raises(
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["read", "get_POS_MSEC"],
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release_gate = threading.Event()
        worker_done = threading.Event()

        capture = GatedThreadedCapture(
            phase=phase,
            on_enter=lambda: loop.call_soon_threadsafe(entered.set),
            release_gate=release_gate,
            worker_done=worker_done,
            native_raises=True,
        )
        install_fake_cv2(monkeypatch, FakeCv2(capture))

        is_live = phase == "read"
        config = StreamConfig(
            stream_id=f"stream-err-{phase}",
            camera_external_id=f"cam-err-{phase}",
            source_url="rtsp://camera.local/live" if is_live else r"D:\data\clip.mp4",
            is_live=is_live,
        )
        source = OpenCvFrameSource(config)
        await source.connect()

        read_task: asyncio.Task[object] | None = None
        close_task: asyncio.Task[None] | None = None
        try:
            read_task = asyncio.create_task(source.read_frame())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert capture.busy is True

            close_task = asyncio.create_task(source.close())
            await spin()

            read_task.cancel()
            await spin()
            read_task.cancel()
            await spin()
            assert capture.busy is True
            assert not close_task.done()
            assert capture.release_calls == 0
            assert not capture.overlap_detected

            release_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read_task, timeout=1.0)
            await asyncio.wait_for(close_task, timeout=1.0)

            assert capture.busy is False
            assert capture.release_calls == 1
            assert not capture.overlap_detected
            assert source.is_connected is False
        finally:
            release_gate.set()
            await asyncio.to_thread(worker_done.wait, 1.0)
            if read_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(read_task, timeout=1.0)
            if close_task is not None:
                with suppress(BaseException):
                    await asyncio.wait_for(close_task, timeout=1.0)
            with suppress(BaseException):
                await asyncio.wait_for(source.close(), timeout=1.0)

    asyncio.run(scenario())
