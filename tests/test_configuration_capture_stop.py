"""Operator-local stop requests; no native source, model or network server."""

import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartsite_ai.tools import run_configuration_capture as cli
from smartsite_ai.tools.configuration_host_config import ConfigurationHostError


def isolate_signals(monkeypatch, calls):
    def register(*_):
        calls.append("signal")
        return signal.SIG_DFL

    names = ("SIGINT", "SIGTERM", "SIGBREAK")
    values = {name: getattr(signal, name) for name in names if hasattr(signal, name)}
    monkeypatch.setattr(cli, "signal", SimpleNamespace(signal=register, **values))


def test_cli_accepts_absent_same_directory_stop_file(tmp_path, monkeypatch):
    manifest, stop = tmp_path / "host.json", tmp_path / "STOP"
    config, calls = object(), []
    monkeypatch.setattr(cli, "load_host_configuration", lambda *_: config)

    async def command(value, *, stop_file=None):
        calls.append((value, stop_file))

    monkeypatch.setattr(cli, "_command", command)
    assert cli.main(["--manifest", str(manifest), "--stop-file", str(stop)]) == 0
    assert calls == [(config, stop)]


@pytest.mark.parametrize("kind", ["relative", "different_parent", "wrong_name", "existing"])
def test_cli_refuses_unsafe_or_preexisting_stop_before_command(tmp_path, monkeypatch, capsys, kind):
    manifest, stop = tmp_path / "host.json", tmp_path / "STOP"
    if kind == "relative":
        stop = Path("rtsp://private:credential@example.invalid/STOP")
    elif kind == "different_parent":
        stop = tmp_path.parent / "STOP"
    elif kind == "wrong_name":
        stop = tmp_path / "other"
    else:
        stop.write_text("Operator already requested stop")
    calls = []
    monkeypatch.setattr(cli, "load_host_configuration", lambda *_: object())

    async def command(*args, **kwargs):
        calls.append(True)

    monkeypatch.setattr(cli, "_command", command)
    assert cli.main(["--manifest", str(manifest), "--stop-file", str(stop)]) == 1
    assert not calls
    output = capsys.readouterr()
    assert "CONFIGURATION_HOST_INVALID_CONFIG" in output.out
    assert "private:credential" not in output.out + output.err


def test_cli_without_stop_file_preserves_existing_command_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "load_host_configuration", lambda *_: object())

    async def command(config):
        calls.append(config)

    monkeypatch.setattr(cli, "_command", command)
    assert cli.main(["--manifest", str(tmp_path / "host.json")]) == 0
    assert len(calls) == 1


def test_stop_file_sets_existing_event_without_cancelling_host(tmp_path, monkeypatch):
    stop, calls = tmp_path / "STOP", []
    isolate_signals(monkeypatch, [])

    async def host(config, *, stop_requested):
        calls.append("started")
        stop.write_text("Stop this owned source")
        await asyncio.wait_for(stop_requested.wait(), timeout=1)
        calls.append("released")
        return {}

    monkeypatch.setattr(cli, "serve_host", host)
    asyncio.run(cli._command(object(), stop_file=stop))
    assert calls == ["started", "released"]


def test_stop_watch_io_failure_requests_drain_and_returns_safe_error(tmp_path, monkeypatch):
    stop, calls = tmp_path / "STOP", []
    isolate_signals(monkeypatch, calls)

    def unreadable(self):
        raise PermissionError("private source locator must not escape")

    monkeypatch.setattr(Path, "exists", unreadable)

    async def host(config, *, stop_requested):
        await asyncio.wait_for(stop_requested.wait(), timeout=1)
        calls.append("drained")
        return {}

    monkeypatch.setattr(cli, "serve_host", host)
    with pytest.raises(ConfigurationHostError, match="CONFIGURATION_HOST_INVALID_CONFIG"):
        asyncio.run(cli._command(object(), stop_file=stop))
    assert "drained" in calls
    assert calls.count("signal") % 2 == 0


def test_command_cancellation_cleans_watch_and_restores_signals(tmp_path, monkeypatch):
    calls = []
    isolate_signals(monkeypatch, calls)

    async def scenario():
        started = asyncio.Event()

        async def host(config, *, stop_requested):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                calls.append("host-finally")

        monkeypatch.setattr(cli, "serve_host", host)
        task = asyncio.create_task(cli._command(object(), stop_file=tmp_path / "STOP"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(t is asyncio.current_task() or t.done() for t in asyncio.all_tasks())

    asyncio.run(scenario())
    assert "host-finally" in calls
    assert calls.count("signal") % 2 == 0
