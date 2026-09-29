from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.source import SourceConnectionError
from smartsite_ai.tools.probe_source import (
    SourceProbeError,
    execute_probe,
    opencv_source_factory,
    run_probe,
)

_SECRET = "rtsp://operator:s3cret-token@10.0.0.8/stream1"
_ENV = "SMARTSITE_AI_CAMERA_TAPO_SOURCE"


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _frame(width: int = 2, height: int = 1, sequence: int = 0) -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="source-probe",
        session_id=uuid4(),
        camera_external_id="source-probe",
        captured_at=datetime(2026, 9, 29, tzinfo=UTC),
        width=width,
        height=height,
        sequence_number=sequence,
        payload=b"\x00" * (width * height * 3),
    )


class FakeSource:
    def __init__(
        self,
        frames: list[FrameEnvelope | None] | None = None,
        *,
        connect_error: BaseException | None = None,
        clock: Clock | None = None,
        advance_on_connect: float | None = None,
    ) -> None:
        self.frames = list(frames or [])
        self.connect_error = connect_error
        self.clock = clock
        self.advance_on_connect = advance_on_connect
        self.connect_calls = 0
        self.read_calls = 0
        self.close_calls = 0
        self.connected = False

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.advance_on_connect is not None and self.clock is not None:
            self.clock.now = self.advance_on_connect
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    async def read_frame(self) -> FrameEnvelope | None:
        self.read_calls += 1
        if not self.frames:
            return None
        return self.frames.pop(0)

    async def close(self) -> None:
        self.close_calls += 1
        self.connected = False


def _factory(source: FakeSource):
    def factory(config: object) -> FakeSource:
        factory.configs.append(config)
        return source

    factory.configs = []
    return factory


def _run(
    argv: list[str],
    environ: dict[str, str] | None = None,
    *,
    source: FakeSource | None = None,
    clock: Clock | None = None,
    wait_for=None,
) -> tuple[int, str, str]:
    outputs: list[str] = []
    errors: list[str] = []

    def emit(line: str) -> None:
        outputs.append(line)

    code = run_probe(
        argv,
        environ or {},
        source_factory=None if source is None else _factory(source),
        clock=clock,
        wait_for=wait_for,
        emit=emit,
        emit_error=errors.append,
    )
    return code, "\n".join(outputs), "\n".join(errors)


def test_probe_reads_bounded_frames_and_hides_the_source() -> None:
    source = FakeSource([_frame(4, 2, 0), _frame(4, 2, 1), _frame(9, 9, 2)])
    clock = Clock(10.0)

    def clock_at_end() -> float:
        if source.read_calls >= 2:
            clock.now = 10.25
        return clock.now

    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "2", "--timeout-seconds", "15"],
        {_ENV: _SECRET},
        source=source,
        clock=clock_at_end,
    )

    assert code == 0
    assert err == ""
    assert json.loads(out) == {
        "status": "ok",
        "framesRead": 2,
        "width": 4,
        "height": 2,
        "elapsedMs": 250,
    }
    assert _SECRET not in out
    assert "s3cret-token" not in out
    assert _ENV not in out
    assert source.connect_calls == 1
    assert source.read_calls == 2
    assert source.close_calls == 1
    assert source.connected is False


def test_eof_after_one_frame_is_success() -> None:
    source = FakeSource([_frame()])
    code, out, _err = _run(
        ["--source-env", _ENV, "--max-frames", "30", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
    )
    assert code == 0
    assert json.loads(out)["framesRead"] == 1
    assert source.close_calls == 1


@pytest.mark.parametrize(
    ("environ", "argv_env"),
    [
        ({}, _ENV),
        ({_ENV: "   "}, _ENV),
    ],
)
def test_unset_source_fails_closed_without_opening(environ: dict[str, str], argv_env: str) -> None:
    source = FakeSource([_frame()])
    code, out, err = _run(
        ["--source-env", argv_env, "--max-frames", "1", "--timeout-seconds", "1"],
        environ,
        source=source,
    )
    body = json.loads(out)
    assert code == 1
    assert err == ""
    assert body == {
        "error": "SourceProbeError",
        "code": "SOURCE_UNSET",
        "message": "source environment variable is unset",
    }
    assert source.connect_calls == 0
    assert source.close_calls == 0


@pytest.mark.parametrize(
    "source_env",
    [
        "rtsp://operator:s3cret-token@10.0.0.8/stream1",
        "SMARTSITE_AI_",
        "smartsite_ai_camera",
        "SMARTSITE_AI_" + ("A" * 81),
    ],
)
def test_invalid_env_name_or_bounds_are_rejected_without_echo(source_env: str) -> None:
    code, out, err = _run(
        ["--source-env", source_env, "--max-frames", "1", "--timeout-seconds", "1"],
        {source_env: _SECRET},
    )
    assert code == 1
    assert json.loads(out)["code"] == "INVALID_ARGS"
    assert "s3cret-token" not in out
    assert "s3cret-token" not in err
    assert "10.0.0.8" not in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--source-env", _ENV, "--max-frames", "0", "--timeout-seconds", "1"],
        ["--source-env", _ENV, "--max-frames", "301", "--timeout-seconds", "1"],
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "0.9"],
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "121"],
        ["--source-env", _ENV],
        ["--source", _SECRET],
    ],
)
def test_invalid_arguments_use_the_allowlisted_message(argv: list[str]) -> None:
    code, out, err = _run(argv, {_ENV: _SECRET})
    assert code == 1
    assert json.loads(out) == {
        "error": "SourceProbeError",
        "code": "INVALID_ARGS",
        "message": "source probe arguments are invalid",
    }
    assert "s3cret-token" not in out + err


def test_invalid_source_value_does_not_open_or_echo() -> None:
    source = FakeSource([_frame()])
    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "1"],
        {_ENV: "rtsp://operator:s3cret-token@"},
        source=source,
    )
    assert code == 1
    assert json.loads(out)["code"] == "SOURCE_INVALID"
    assert source.connect_calls == 0
    assert "s3cret-token" not in out + err


def test_zero_frames_fail_closed_and_release() -> None:
    source = FakeSource([])
    code, out, _err = _run(
        ["--source-env", _ENV, "--max-frames", "3", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
    )
    assert code == 1
    assert json.loads(out)["code"] == "NO_FRAMES"
    assert source.close_calls == 1


def test_driver_error_is_allowlisted_and_capture_is_released() -> None:
    source = FakeSource(
        connect_error=SourceConnectionError(
            "OpenCV could not open rtsp://operator:s3cret-token@10.0.0.8/stream1"
        )
    )
    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
    )
    assert code == 1
    assert json.loads(out) == {
        "error": "SourceProbeError",
        "code": "SOURCE_UNAVAILABLE",
        "message": "source probe could not read the source",
    }
    assert "s3cret-token" not in out + err
    assert "OpenCV" not in out
    assert source.close_calls == 1


def test_timeout_after_partial_decode_is_not_success() -> None:
    clock = Clock(0.0)
    source = FakeSource([_frame()], clock=clock, advance_on_connect=5.0)
    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "2", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
        clock=clock,
    )
    assert code == 1
    assert json.loads(out)["code"] == "TIMEOUT"
    assert "framesRead" not in out
    assert "s3cret-token" not in out + err
    assert source.read_calls == 0
    assert source.close_calls == 1


def test_wait_timeout_maps_without_the_waiter_message() -> None:
    source = FakeSource([_frame()])

    async def expire(awaitable: object, timeout: float) -> object:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError(f"timed out reading {_SECRET} after {timeout}")

    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
        wait_for=expire,
    )
    assert code == 1
    assert json.loads(out)["code"] == "TIMEOUT"
    assert _SECRET not in out + err
    assert source.close_calls == 1


def test_keyboard_interrupt_releases_and_returns_130() -> None:
    source = FakeSource(connect_error=KeyboardInterrupt())
    code, out, err = _run(
        ["--source-env", _ENV, "--max-frames", "1", "--timeout-seconds", "5"],
        {_ENV: _SECRET},
        source=source,
    )
    assert code == 130
    assert json.loads(out) == {
        "error": "SourceProbeError",
        "code": "INTERRUPTED",
        "message": "source probe was interrupted",
    }
    assert err == ""
    assert source.close_calls == 1


def test_default_factory_builds_a_closed_opencv_source(tmp_path) -> None:
    from smartsite_ai.ingestion.config import StreamConfig

    config = StreamConfig(
        stream_id="source-probe",
        camera_external_id="source-probe",
        source_url=str(tmp_path / "clip.mp4"),
        is_live=True,
    )
    source = opencv_source_factory(config)
    assert isinstance(source, OpenCvFrameSource)
    assert source.is_connected is False


def test_probe_error_string_is_only_the_code() -> None:
    assert str(SourceProbeError("TIMEOUT")) == "TIMEOUT"


def test_cancelled_probe_still_closes() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        class Blocking(FakeSource):
            async def connect(self) -> None:
                await super().connect()
                started.set()
                await asyncio.Future()

        blocking = Blocking([])
        task = asyncio.create_task(
            execute_probe(
                source_env=_ENV,
                max_frames=1,
                timeout_seconds=5,
                environ={_ENV: _SECRET},
                source_factory=_factory(blocking),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert blocking.close_calls == 1

    asyncio.run(scenario())
