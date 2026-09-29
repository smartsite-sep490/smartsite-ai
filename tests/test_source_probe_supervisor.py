from __future__ import annotations

import io
import json
import subprocess

from smartsite_ai.tools.probe_source import _error_line, supervise_probe

_SECRET = "rtsp://operator:s3cret-token@10.0.0.8/stream1"
_ENV = "SMARTSITE_AI_CAMERA_TAPO_SOURCE"
_ARGV = ["--source-env", _ENV, "--max-frames", "2", "--timeout-seconds", "5"]
_SUCCESS = b'{"status":"ok","framesRead":2,"width":4,"height":2,"elapsedMs":250}\n'


class FakeProcess:
    def __init__(
        self,
        stdout: bytes = _SUCCESS,
        stderr: bytes = b"",
        returncode: int = 0,
        *,
        already_exited: bool = True,
        die_on_terminate: bool = True,
        interrupt_wait: bool = False,
    ) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._code = returncode
        self.already_exited = already_exited
        self.die_on_terminate = die_on_terminate
        self.interrupt_wait = interrupt_wait
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        if self.interrupt_wait and not self.terminated and not self.killed:
            raise KeyboardInterrupt
        if self.killed or (self.terminated and self.die_on_terminate) or self.already_exited:
            self.returncode = self._code
            return self._code
        raise subprocess.TimeoutExpired(cmd="probe", timeout=0)

    def terminate(self) -> None:
        self.terminated = True
        if self.die_on_terminate:
            self.returncode = self._code

    def kill(self) -> None:
        self.killed = True
        self.returncode = self._code


def _popen_factory(process: FakeProcess, recorded: dict[str, object]):
    def popen(cmd: list[str], **kwargs: object) -> FakeProcess:
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return process

    return popen


def _run(
    process: FakeProcess,
    argv: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, object]]:
    recorded: dict[str, object] = {}
    lines: list[str] = []
    code = supervise_probe(
        argv or _ARGV,
        environ or {_ENV: _SECRET, "PATH": "C:\\Windows"},
        popen=_popen_factory(process, recorded),
        emit=lines.append,
        allowance_seconds=1.0,
        terminate_grace_seconds=0.5,
    )
    return code, "\n".join(lines), recorded


def test_parent_emits_one_valid_success_line_and_exits_0() -> None:
    code, out, recorded = _run(FakeProcess())
    assert code == 0
    assert json.loads(out) == {
        "status": "ok",
        "framesRead": 2,
        "width": 4,
        "height": 2,
        "elapsedMs": 250,
    }
    cmd = recorded["cmd"]
    assert isinstance(cmd, list)
    assert "--worker" in cmd
    assert _ENV in cmd
    assert "2" in cmd
    assert "5" in cmd
    assert _SECRET not in cmd
    kwargs = recorded["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"][_ENV] == _SECRET
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert _SECRET not in out


def test_parent_preserves_child_failure_exit() -> None:
    line = _error_line("NO_FRAMES")
    code, out, _recorded = _run(FakeProcess(stdout=(line + "\n").encode(), returncode=1))
    assert code == 1
    assert out == line
    assert _SECRET not in out


def test_parent_hides_stderr_and_source_secret() -> None:
    process = FakeProcess(stderr=f"driver {_SECRET}\n".encode())
    code, out, _recorded = _run(process)
    assert code == 0
    assert _SECRET not in out
    assert "driver" not in out


def test_parent_rejects_unsafe_child_output_without_reflecting_it() -> None:
    secret_tail = f"\n{_SECRET}"
    samples = [
        b"not-json " + _SECRET.encode(),
        _SUCCESS + _SECRET.encode() + b"\n",
        b'{"error":"SourceProbeError","code":"EXPLODED","message":"boom"}\n',
        (
            b'{"status":"ok","framesRead":1,"width":1,"height":1,'
            b'"elapsedMs":0,"source":"' + _SECRET.encode() + b'"}\n'
        ),
        b"x" * 5000,
    ]
    for stdout in samples:
        code, out, _recorded = _run(FakeProcess(stdout=stdout, returncode=1))
        assert code == 1
        assert out == _error_line("SOURCE_UNAVAILABLE")
        assert _SECRET not in out
        assert secret_tail.strip() not in out
        assert "EXPLODED" not in out


def test_timeout_terminates_then_kills_and_waits() -> None:
    process = FakeProcess(already_exited=False, die_on_terminate=False, returncode=1)
    code, out, _recorded = _run(process)
    assert code == 1
    assert out == _error_line("TIMEOUT")
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls >= 2
    assert process.returncode is not None
    assert _SECRET not in out


def test_interrupt_terminates_kills_and_waits() -> None:
    process = FakeProcess(
        already_exited=False,
        die_on_terminate=False,
        interrupt_wait=True,
        returncode=130,
    )
    code, out, _recorded = _run(process)
    assert code == 130
    assert out == _error_line("INTERRUPTED")
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls >= 2
    assert process.returncode is not None
    assert _SECRET not in out
