"""Independent-review regressions for configuration capture, using synthetic pixels."""

import asyncio
import importlib
import threading
from datetime import timedelta
from unittest.mock import patch

import pytest
from configuration_capture_fakes import NOW, finish, frame, until
from test_configuration_capture import capture_pair


def test_newer_invalid_timestamp_cannot_resurrect_an_older_inflight_encode():
    async def scenario():
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        image = importlib.import_module("smartsite_ai.configuration_image")

        def encode(value):
            started.set()
            assert release.wait(2)
            result = image.encode_configuration_image(value)
            finished.set()
            return result

        capture, source, _, _ = capture_pair(encoder=encode)
        try:
            await capture.start()
            await source.reading.wait()
            source.items.put_nowait(frame())
            await until(started.is_set)
            source.items.put_nowait(frame(1, captured_at=NOW - timedelta(seconds=3)))
            await until(lambda: source.reads == 3)
            release.set()
            await until(finished.is_set)
            await asyncio.sleep(0.01)
            assert capture.latest() is None
        finally:
            release.set()
            await finish(capture, source)

    asyncio.run(scenario())


@pytest.mark.parametrize("dimensions", [(63, 64), (64, 63), (16384, 16384)])
def test_pixel_budget_is_checked_before_pillow_allocates(dimensions):
    module = importlib.import_module("smartsite_ai.configuration_image")
    # Only dimensions are relevant to this early guard: avoid a 768MiB fixture.
    raw = frame().model_copy(update={"width": dimensions[0], "height": dimensions[1]})
    with (
        patch("PIL.Image.frombytes", side_effect=AssertionError("allocation reached")),
        pytest.raises(ValueError, match="input budget"),
    ):
        module.encode_configuration_image(raw)


def test_unexpected_owner_task_cancellation_never_proves_native_handle_release():
    async def scenario():
        capture, source, lease, module = capture_pair()
        await capture.start()
        await source.reading.wait()
        # Failure injection only. Public lifecycle never cancels this task: a
        # native I/O thread may still be running after its coroutine is cancelled.
        capture._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await capture._task
        assert capture.state == "FAILED"
        assert capture.reason == "capture_cancelled"
        assert capture.latest() is None
        assert source.closes == 0
        assert not await capture.stop()
        with pytest.raises(RuntimeError):
            await module.ConfigurationCapture(lease).start()
        await source.close()  # synthetic fixture teardown, not an owner recovery claim

    asyncio.run(scenario())
