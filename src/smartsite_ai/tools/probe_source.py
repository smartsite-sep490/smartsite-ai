"""Check one camera or video source before the MF05/MF06 runtime starts."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from pydantic import ValidationError

from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource

_ENV_NAME_RE = re.compile(r"^SMARTSITE_AI_[A-Z0-9_]{1,80}$")
_MESSAGES = {
    "INVALID_ARGS": "source probe arguments are invalid",
    "SOURCE_UNSET": "source environment variable is unset",
    "SOURCE_INVALID": "source environment value is invalid",
    "NO_FRAMES": "source produced no frames",
    "TIMEOUT": "source probe timed out",
    "SOURCE_UNAVAILABLE": "source probe could not read the source",
    "INTERRUPTED": "source probe was interrupted",
}
_T = TypeVar("_T")
_WaitFor = Callable[[Awaitable[_T], float], Awaitable[_T]]
_PROCESS_ALLOWANCE_SECONDS = 1.0
_TERMINATE_GRACE_SECONDS = 0.5
_MAX_CHILD_OUTPUT_BYTES = 4096
_PopenFactory = Callable[..., object]


class SourceProbeError(Exception):
    """Allowlisted probe failure. Its text is only the stable code."""

    def __init__(self, code: str) -> None:
        if code not in _MESSAGES:
            code = "SOURCE_UNAVAILABLE"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProbeResult:
    frames_read: int
    width: int
    height: int
    elapsed_ms: int


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SourceProbeError("INVALID_ARGS")


def opencv_source_factory(config: StreamConfig) -> OpenCvFrameSource:
    """Build one unopened OpenCV source for a validated stream config."""

    return OpenCvFrameSource(config)


def run_probe(
    argv: list[str],
    environ: Mapping[str, str],
    *,
    source_factory: Callable[[StreamConfig], object] | None = None,
    clock: Callable[[], float] | None = None,
    wait_for: _WaitFor[object] | None = None,
    emit: Callable[[str], None] | None = None,
    emit_error: Callable[[str], None] | None = None,
) -> int:
    """Run the probe and emit one safe JSON line. Returns 0, 1, or 130."""

    del emit_error
    writer = emit or print
    try:
        source_env, max_frames, timeout_seconds = _parse_args(argv)
        result = asyncio.run(
            execute_probe(
                source_env=source_env,
                max_frames=max_frames,
                timeout_seconds=timeout_seconds,
                environ=environ,
                source_factory=source_factory,
                clock=clock,
                wait_for=wait_for,
            )
        )
    except SourceProbeError as exc:
        writer(_error_line(exc.code))
        return 130 if exc.code == "INTERRUPTED" else 1
    except KeyboardInterrupt:
        writer(_error_line("INTERRUPTED"))
        return 130
    except Exception:
        writer(_error_line("SOURCE_UNAVAILABLE"))
        return 1
    writer(_success_line(result))
    return 0


async def execute_probe(
    *,
    source_env: str,
    max_frames: int,
    timeout_seconds: float,
    environ: Mapping[str, str],
    source_factory: Callable[[StreamConfig], object] | None = None,
    clock: Callable[[], float] | None = None,
    wait_for: _WaitFor[object] | None = None,
) -> ProbeResult:
    """Resolve one env source, read at most max_frames, and always close it."""

    frames_bound, timeout_bound = _validate_bounds(max_frames, timeout_seconds)
    source_value = _resolve_source(source_env, environ)
    config = _stream_config(source_value)
    factory = source_factory or opencv_source_factory
    source = factory(config)
    try:
        return await _read_probe(
            source,
            max_frames=frames_bound,
            timeout_seconds=timeout_bound,
            clock=clock or monotonic,
            wait_for=wait_for or asyncio.wait_for,
        )
    except KeyboardInterrupt:
        raise SourceProbeError("INTERRUPTED") from None
    except asyncio.CancelledError:
        raise
    except SourceProbeError:
        raise
    except Exception:
        raise SourceProbeError("SOURCE_UNAVAILABLE") from None
    finally:
        await _release(source)


def supervise_probe(
    argv: list[str],
    environ: Mapping[str, str],
    *,
    popen: _PopenFactory | None = None,
    executable: str | None = None,
    emit: Callable[[str], None] | None = None,
    allowance_seconds: float = _PROCESS_ALLOWANCE_SECONDS,
    terminate_grace_seconds: float = _TERMINATE_GRACE_SECONDS,
) -> int:
    """Run the probe in a child process and stop that process at a hard deadline.

    The child inherits ``environ``. Its command carries the variable name and numeric
    bounds only. ``allowance_seconds`` is extra startup time after ``timeout_seconds``.
    A forced stop terminates, kills if needed, and waits; it does not claim that the
    worker closed its capture.
    """

    writer = emit or print
    try:
        source_env, max_frames, timeout_seconds = _parse_args(argv)
    except SourceProbeError as exc:
        writer(_error_line(exc.code))
        return 1
    command = _worker_command(
        executable or sys.executable,
        source_env,
        max_frames,
        timeout_seconds,
    )
    launcher = popen or subprocess.Popen
    try:
        process = launcher(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environ),
            creationflags=_creationflags(),
        )
    except Exception:
        writer(_error_line("SOURCE_UNAVAILABLE"))
        return 1
    budget = timeout_seconds + allowance_seconds
    try:
        process.wait(timeout=budget)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        _halt_process(process, terminate_grace_seconds)
        writer(_error_line("INTERRUPTED"))
        return 130
    except subprocess.TimeoutExpired:
        _halt_process(process, terminate_grace_seconds)
        writer(_error_line("TIMEOUT"))
        return 1
    stdout, _stderr = _drain(process)
    line, code = _validated_child_output(stdout, _returncode(process))
    writer(line)
    return code


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--worker"]:
        raise SystemExit(run_probe(args[1:], os.environ))
    raise SystemExit(supervise_probe(args, os.environ))


def _parse_args(argv: list[str]) -> tuple[str, int, float]:
    parser = _Parser(prog="smartsite-ai-source-probe", exit_on_error=False)
    parser.add_argument("--source-env", required=True)
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SourceProbeError, SystemExit):
        raise SourceProbeError("INVALID_ARGS") from None
    return args.source_env, args.max_frames, args.timeout_seconds


def _validate_bounds(max_frames: int, timeout_seconds: float) -> tuple[int, float]:
    if isinstance(max_frames, bool) or not isinstance(max_frames, int):
        raise SourceProbeError("INVALID_ARGS")
    if not 1 <= max_frames <= 300:
        raise SourceProbeError("INVALID_ARGS")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 1 <= float(timeout_seconds) <= 120
    ):
        raise SourceProbeError("INVALID_ARGS")
    return max_frames, float(timeout_seconds)


def _resolve_source(source_env: str, environ: Mapping[str, str]) -> str:
    if not isinstance(source_env, str) or _ENV_NAME_RE.fullmatch(source_env) is None:
        raise SourceProbeError("INVALID_ARGS")
    value = environ.get(source_env)
    if not isinstance(value, str) or value.strip() == "":
        raise SourceProbeError("SOURCE_UNSET")
    return value.strip()


def _stream_config(source_value: str) -> StreamConfig:
    try:
        return StreamConfig(
            stream_id="source-probe",
            camera_external_id="source-probe",
            source_url=source_value,
            is_live=True,
            pace_replay=False,
        )
    except ValidationError:
        raise SourceProbeError("SOURCE_INVALID") from None


async def _read_probe(
    source: object,
    *,
    max_frames: int,
    timeout_seconds: float,
    clock: Callable[[], float],
    wait_for: _WaitFor[object],
) -> ProbeResult:
    origin = clock()
    deadline = origin + timeout_seconds
    connect = getattr(source, "connect", None)
    read_frame = getattr(source, "read_frame", None)
    if connect is None or read_frame is None:
        raise SourceProbeError("SOURCE_UNAVAILABLE")

    await _bounded(connect(), deadline, clock, wait_for)
    frames_read = 0
    width = 0
    height = 0
    while frames_read < max_frames:
        frame = await _bounded(read_frame(), deadline, clock, wait_for)
        if frame is None:
            break
        if frames_read == 0:
            width, height = _frame_size(frame)
        frames_read += 1
    if frames_read == 0:
        raise SourceProbeError("NO_FRAMES")
    elapsed_ms = int(round((clock() - origin) * 1000))
    return ProbeResult(
        frames_read=frames_read,
        width=width,
        height=height,
        elapsed_ms=max(elapsed_ms, 0),
    )


def _frame_size(frame: object) -> tuple[int, int]:
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if isinstance(width, bool) or isinstance(height, bool):
        raise SourceProbeError("SOURCE_UNAVAILABLE")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise SourceProbeError("SOURCE_UNAVAILABLE")
    return width, height


async def _bounded(
    awaitable: Awaitable[object],
    deadline: float,
    clock: Callable[[], float],
    wait_for: _WaitFor[object],
) -> object:
    remaining = deadline - clock()
    if remaining <= 0:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise SourceProbeError("TIMEOUT")
    try:
        return await wait_for(awaitable, remaining)
    except TimeoutError:
        raise SourceProbeError("TIMEOUT") from None
    except (KeyboardInterrupt, asyncio.CancelledError, SourceProbeError):
        raise
    except Exception:
        raise SourceProbeError("SOURCE_UNAVAILABLE") from None


async def _release(source: object) -> None:
    close = getattr(source, "close", None)
    if close is None:
        return
    task = asyncio.current_task()
    suspended = 0
    if task is not None:
        while task.cancelling():
            task.uncancel()
            suspended += 1
    try:
        await close()
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise
    except Exception:
        return
    finally:
        if task is not None:
            for _ in range(suspended):
                task.cancel()


def _worker_command(
    executable: str,
    source_env: str,
    max_frames: int,
    timeout_seconds: float,
) -> list[str]:
    return [
        executable,
        "-m",
        "smartsite_ai.tools.probe_source",
        "--worker",
        "--source-env",
        source_env,
        "--max-frames",
        str(max_frames),
        "--timeout-seconds",
        format(timeout_seconds, "g"),
    ]


def _creationflags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _halt_process(process: object, grace_seconds: float) -> None:
    terminate = getattr(process, "terminate", None)
    kill = getattr(process, "kill", None)
    if terminate is not None:
        with suppress(Exception):
            terminate()
    if _wait_for_exit(process, grace_seconds):
        _drain(process)
        return
    if kill is not None:
        with suppress(Exception):
            kill()
    if not _wait_for_exit(process, grace_seconds):
        wait = getattr(process, "wait", None)
        if wait is not None:
            try:
                wait()
            except KeyboardInterrupt:
                if kill is not None:
                    kill()
                wait()
    _drain(process)


def _wait_for_exit(process: object, grace_seconds: float) -> bool:
    wait = getattr(process, "wait", None)
    if wait is None:
        return False
    try:
        wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return False
    except KeyboardInterrupt:
        return False
    return True


def _drain(process: object) -> tuple[bytes, bytes]:
    return _read_pipe(getattr(process, "stdout", None)), _read_pipe(
        getattr(process, "stderr", None)
    )


def _read_pipe(pipe: object) -> bytes:
    if pipe is None:
        return b""
    read = getattr(pipe, "read", None)
    if read is None:
        return b""
    try:
        chunk = read()
    except Exception:
        return b""
    return chunk if isinstance(chunk, bytes) else b""


def _returncode(process: object) -> int | None:
    code = getattr(process, "returncode", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _validated_child_output(stdout: bytes, returncode: int | None) -> tuple[str, int]:
    payload = _child_payload(stdout)
    if payload is None or returncode is None:
        return _error_line("SOURCE_UNAVAILABLE"), 1
    keys = set(payload)
    if keys == {"status", "framesRead", "width", "height", "elapsedMs"}:
        result = _success_payload(payload)
        if result is None or returncode != 0:
            return _error_line("SOURCE_UNAVAILABLE"), 1
        return _success_line(result), 0
    if keys == {"error", "code", "message"}:
        code = payload.get("code")
        if (
            payload.get("error") != "SourceProbeError"
            or not isinstance(code, str)
            or payload.get("message") != _MESSAGES.get(code)
        ):
            return _error_line("SOURCE_UNAVAILABLE"), 1
        expected = 130 if code == "INTERRUPTED" else 1
        if returncode != expected:
            return _error_line("SOURCE_UNAVAILABLE"), 1
        return _error_line(code), expected
    return _error_line("SOURCE_UNAVAILABLE"), 1


def _child_payload(stdout: bytes) -> dict[str, object] | None:
    if len(stdout) > _MAX_CHILD_OUTPUT_BYTES or b"\x00" in stdout:
        return None
    try:
        text = stdout.decode("utf-8")
    except UnicodeError:
        return None
    lines = text.splitlines()
    if len(lines) != 1:
        return None
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _success_payload(payload: Mapping[str, object]) -> ProbeResult | None:
    frames_read = payload.get("framesRead")
    width = payload.get("width")
    height = payload.get("height")
    elapsed_ms = payload.get("elapsedMs")
    if payload.get("status") != "ok":
        return None
    if not _bounded_int(frames_read, 1, 300):
        return None
    if not _bounded_int(width, 1, 16_384):
        return None
    if not _bounded_int(height, 1, 16_384):
        return None
    if not _bounded_int(elapsed_ms, 0, 3_600_000):
        return None
    assert isinstance(frames_read, int)
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(elapsed_ms, int)
    return ProbeResult(
        frames_read=frames_read,
        width=width,
        height=height,
        elapsed_ms=elapsed_ms,
    )


def _bounded_int(value: object, lower: int, upper: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper


def _success_line(result: ProbeResult) -> str:
    return json.dumps(
        {
            "status": "ok",
            "framesRead": result.frames_read,
            "width": result.width,
            "height": result.height,
            "elapsedMs": result.elapsed_ms,
        },
        separators=(",", ":"),
    )


def _error_line(code: str) -> str:
    if code not in _MESSAGES:
        code = "SOURCE_UNAVAILABLE"
    return json.dumps(
        {"error": "SourceProbeError", "code": code, "message": _MESSAGES[code]},
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
