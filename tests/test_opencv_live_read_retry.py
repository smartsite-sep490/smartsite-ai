"""Opt-in retry on one open capture; no camera hardware, model, or reconnect."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest
from test_opencv_source import FakeCapture, FakeCv2, install_fake_cv2, make_bgr_frame, make_config

from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.source import SourceReadError
from smartsite_ai.integrations.worker_generation import (
    GenerationBoundSource,
    WorkerGenerationCredential,
)


class ScriptedCapture(FakeCapture):
    def __init__(self, reads: list[tuple[bool, object | None]]) -> None:
        super().__init__()
        self.reads = reads

    def read(self) -> tuple[bool, object | None]:
        self.read_calls += 1
        return self.reads.pop(0) if self.reads else (False, None)


def test_transient_empty_live_reads_keep_one_capture_session_and_real_frame_sequence(monkeypatch):
    async def run():
        capture = ScriptedCapture(
            [
                (True, make_bgr_frame(2, 2, 1)),
                (False, None),
                (True, None),
                (True, make_bgr_frame(2, 2, 2)),
            ]
        )
        cv2 = FakeCv2(lambda _: capture)
        install_fake_cv2(monkeypatch, cv2)
        waits = []

        async def sleep(delay):
            waits.append(delay)

        source = OpenCvFrameSource(make_config("0"), live_read_attempts=3, sleeper=sleep)
        await source.connect()
        try:
            first, second = await source.read_frame(), await source.read_frame()
            assert first is not None and second is not None
            assert first.session_id == second.session_id
            assert (first.sequence_number, second.sequence_number) == (0, 1)
            assert first.payload == bytes([1]) * 12 and second.payload == bytes([2]) * 12
            assert waits == [0.1, 0.1]
            assert cv2.video_capture_inputs == [0]
            assert capture.read_calls == 4 and capture.release_calls == 0
        finally:
            await source.close()
        assert capture.release_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "live,attempts,expected_reads", [(True, 1, 1), (True, 3, 3), (False, 3, 1)]
)
def test_default_live_exhaustion_and_replay_eof_remain_bounded(
    monkeypatch, live, attempts, expected_reads
):
    async def run():
        capture = ScriptedCapture([])
        install_fake_cv2(monkeypatch, FakeCv2(lambda _: capture))
        waits = []

        async def sleep(delay):
            waits.append(delay)

        config = make_config("0").model_copy(update={"is_live": live})
        kwargs = {} if attempts == 1 else {"live_read_attempts": attempts}
        source = OpenCvFrameSource(config, sleeper=sleep, **kwargs)
        await source.connect()
        try:
            assert await source.read_frame() is None
            assert capture.read_calls == expected_reads
            assert len(waits) == expected_reads - 1
        finally:
            await source.close()

    asyncio.run(run())


def test_cancel_during_live_backoff_releases_lock_without_another_read(monkeypatch):
    async def run():
        capture = ScriptedCapture([])
        install_fake_cv2(monkeypatch, FakeCv2(lambda _: capture))
        entered = asyncio.Event()

        async def sleep(_):
            entered.set()
            await asyncio.Event().wait()

        source = OpenCvFrameSource(make_config("0"), live_read_attempts=3, sleeper=sleep)
        await source.connect()
        read = asyncio.create_task(source.read_frame())
        await asyncio.wait_for(entered.wait(), 2)
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        await asyncio.wait_for(source.close(), 2)
        assert capture.read_calls == 1 and capture.release_calls == 1

    asyncio.run(run())


def test_live_retry_survives_the_generation_binding_without_reconnect(monkeypatch):
    async def run():
        capture = ScriptedCapture(
            [(False, None), (True, make_bgr_frame(1, 1)), (True, make_bgr_frame(1, 1))]
        )
        install_fake_cv2(monkeypatch, FakeCv2(lambda _: capture))

        async def sleep(_):
            pass

        credential = WorkerGenerationCredential.model_validate(
            {
                "schemaVersion": "1.0.0",
                "cameraId": "11111111-1111-4111-8111-111111111111",
                "workerGenerationId": "22222222-2222-4222-8222-222222222222",
                "streamSessionId": "33333333-3333-4333-8333-333333333333",
                "secret": "x" * 43,
                "expiresAt": "2026-12-01T00:00:00.000Z",
            }
        )
        source = GenerationBoundSource(
            OpenCvFrameSource(make_config("0"), live_read_attempts=3, sleeper=sleep),
            credential,
            clock=lambda: datetime(2026, 11, 21, tzinfo=UTC),
        )
        await source.connect()
        try:
            first, second = await source.read_frame(), await source.read_frame()
            assert first is not None and second is not None
            assert first.session_id == second.session_id == UUID(credential.stream_session_id)
            assert (first.sequence_number, second.sequence_number) == (0, 1)
            assert capture.read_calls == 3
        finally:
            await source.close()
        with pytest.raises(RuntimeError, match="trusted renewal"):
            await source.connect()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["exception", "invalid-frame"])
def test_native_errors_and_corrupt_frames_are_not_retried(monkeypatch, failure):
    class FailedCapture(ScriptedCapture):
        def read(self):
            if failure == "exception":
                self.read_calls += 1
                raise OSError("synthetic failure")
            return super().read()

    async def run():
        capture = FailedCapture([(True, object())])
        install_fake_cv2(monkeypatch, FakeCv2(lambda _: capture))
        source = OpenCvFrameSource(make_config("0"), live_read_attempts=3)
        await source.connect()
        try:
            with pytest.raises(SourceReadError):
                await source.read_frame()
            assert capture.read_calls == 1
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("attempts", [0, 6, True, 1.5, "3", None])
def test_live_attempt_budget_rejects_invalid_values(attempts):
    with pytest.raises(ValueError):
        OpenCvFrameSource(make_config("0"), live_read_attempts=attempts)


@pytest.mark.parametrize("delay", [-0.1, 0.0, 1.01, float("nan"), float("inf"), True, "0.1"])
def test_live_backoff_rejects_invalid_values(delay):
    with pytest.raises(ValueError):
        OpenCvFrameSource(make_config("0"), live_read_retry_delay=delay)
