import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from smartsite_ai import realtime
from smartsite_ai.config import Settings


def test_realtime_model_loading_does_not_run_on_api_event_loop(monkeypatch) -> None:
    api_thread = threading.get_ident()
    loader_threads = []

    def load(settings):
        loader_threads.append(threading.get_ident())
        raise ValueError("unavailable test artifact")

    monkeypatch.setattr(realtime, "_load_realtime_stack", load)
    socket = MagicMock()
    socket.accept = AsyncMock()
    socket.send_json = AsyncMock()
    socket.close = AsyncMock()

    asyncio.run(realtime.stream_realtime(socket, Settings(realtime_source="0")))

    assert len(loader_threads) == 1
    assert loader_threads[0] != api_thread
    socket.send_json.assert_awaited_once_with(
        {"type": "error", "message": "Realtime model is unavailable"}
    )
    socket.close.assert_awaited_once()


@pytest.mark.parametrize("load_fails", [False, True])
def test_cancelled_load_releases_late_model_and_preserves_cancellation(
    monkeypatch, load_fails
) -> None:
    loader = getattr(realtime, "_load_realtime_stack_async", None)
    assert callable(loader), "background model initialization needs cancellation-safe ownership"
    release = threading.Event()
    runner = MagicMock()

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def load(settings):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(2):
                raise RuntimeError("test loader was not released")
            if load_fails:
                raise ValueError("test artifact failed")
            return (None, runner, None, None)

        monkeypatch.setattr(realtime, "_load_realtime_stack", load)
        task = asyncio.create_task(loader(Settings()))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "cancellation must wait for ownership of the loaded model"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert runner.close.call_count == (0 if load_fails else 1)


def test_repeated_cancellation_still_releases_the_late_model(monkeypatch) -> None:
    release = threading.Event()
    runner = MagicMock()

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        closed = asyncio.Event()
        runner.close.side_effect = closed.set

        def load(settings):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(2):
                raise RuntimeError("test loader was not released")
            return (None, runner, None, None)

        monkeypatch.setattr(realtime, "_load_realtime_stack", load)
        task = asyncio.create_task(realtime._load_realtime_stack_async(Settings()))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            runner.close.assert_not_called()
        finally:
            release.set()
        await asyncio.wait_for(closed.wait(), 1)

    asyncio.run(run())
    runner.close.assert_called_once()
