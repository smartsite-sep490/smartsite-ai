"""Explicit host orchestration; fake capture/server and local ephemeral socket."""

import asyncio
import importlib
import importlib.util
import json
import socket

import pytest


def api():
    name = "smartsite_ai.tools.run_configuration_capture"
    assert importlib.util.find_spec(name) is not None, "Explicit operator host is missing"
    return importlib.import_module(name)


def configuration(tmp_path, **changes):
    config = {
        "schemaVersion": "1.0.0",
        "streamId": "configuration-only",
        "cameraExternalId": "CAMERA-TEST",
        "sourceKind": "LIVE",
        "sourceEnv": "SMARTSITE_AI_CAPTURE_SOURCE",
        "tokenEnv": "SMARTSITE_AI_CAPTURE_TOKEN",
        "resourceId": "e98f6475-0660-4c71-a466-4c93d1739ee0",
        "lockDirectory": str(tmp_path),
        "port": 0,
        "durationSeconds": 0.05,
    }
    config.update(changes)
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config))
    module = importlib.import_module("smartsite_ai.tools.configuration_host_config")
    return module.load_host_configuration(
        path,
        {
            "SMARTSITE_AI_CAPTURE_SOURCE": "0",
            "SMARTSITE_AI_CAPTURE_TOKEN": "synthetic-capture-service-token-12345",
        },
    )


class Runtime:
    def __init__(self, calls, *, released=True, start_failure=False):
        self.calls, self.released, self.start_failure = calls, released, start_failure
        self.state, self.app = "IDLE", object()
        self.reason = None

    async def start(self):
        self.calls.append("start")
        if self.start_failure:
            raise RuntimeError("source://private-credential")
        self.state = "RUNNING"

    async def stop(self):
        self.calls.append("stop")
        self.state = "STOPPED" if self.released else "STOPPING"
        return self.released

    async def wait_for_release(self):
        self.calls.append("drain")
        return self.released


class Server:
    def __init__(self, calls, *, fail=False, wait=False):
        self.calls, self.fail, self.wait = calls, fail, wait
        self.should_exit = False

    async def serve(self, *, sockets):
        self.calls.append("serve")
        assert len(sockets) == 1
        assert sockets[0].getsockname()[0] == "127.0.0.1"
        assert sockets[0].getsockname()[1] > 0
        if self.fail:
            raise RuntimeError("server-secret")
        while self.wait and not self.should_exit:
            await asyncio.sleep(0.001)


def test_normal_host_owns_one_source_and_closes_bound_socket(tmp_path):
    module = api()
    calls, events = [], []
    runtime, server = Runtime(calls), Server(calls)
    result = asyncio.run(
        module.serve_host(
            configuration(tmp_path),
            runtime_factory=lambda _: runtime,
            server_factory=lambda _: server,
            emit=events.append,
        )
    )
    assert calls == ["start", "serve", "stop"]
    assert result["nativeReleaseProved"] is True
    assert events[0]["state"] == "CONFIGURATION_HOST_BOUND_NOT_READY"
    assert events[-1]["state"] == "CONFIGURATION_HOST_STOPPED"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", result["port"]))


def test_occupied_port_refuses_before_runtime_or_source_factory(tmp_path):
    module = api()
    calls = []
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        cfg = configuration(tmp_path, port=occupied.getsockname()[1])
        with pytest.raises(module.ConfigurationHostError) as error:
            asyncio.run(module.serve_host(cfg, runtime_factory=lambda _: calls.append("factory")))
    assert error.value.code == "CONFIGURATION_HOST_BIND_UNAVAILABLE"
    assert calls == []


@pytest.mark.parametrize("kind", ["start", "serve"])
def test_failures_still_stop_once_and_do_not_expose_exception(tmp_path, kind):
    module = api()
    calls, events = [], []
    runtime = Runtime(calls, start_failure=kind == "start")
    server = Server(calls, fail=kind == "serve")
    with pytest.raises(module.ConfigurationHostError) as error:
        asyncio.run(
            module.serve_host(
                configuration(tmp_path),
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: server,
                emit=events.append,
            )
        )
    assert calls.count("stop") == 1
    assert "private-credential" not in str(error.value) and "server-secret" not in str(error.value)
    assert events[-1]["terminationReason"] == "host_failed"
    assert "private-credential" not in json.dumps(events)


def test_unresolved_native_stop_is_not_reported_as_release(tmp_path):
    module = api()
    calls, events = [], []
    runtime = Runtime(calls, released=False)
    with pytest.raises(module.ConfigurationHostError) as error:
        asyncio.run(
            module.serve_host(
                configuration(tmp_path),
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: Server(calls),
                emit=events.append,
            )
        )
    assert error.value.code == "CONFIGURATION_HOST_STOP_UNRESOLVED"
    assert not any(e.get("nativeReleaseProved") is True for e in events)


def test_deadline_requests_server_exit_and_releases_source(tmp_path):
    module = api()
    calls = []
    server = Server(calls, wait=True)
    runtime = Runtime(calls)
    result = asyncio.run(
        module.serve_host(
            configuration(tmp_path),
            runtime_factory=lambda _: runtime,
            server_factory=lambda _: server,
            emit=lambda _: None,
        )
    )
    assert server.should_exit and result["nativeReleaseProved"]
    assert calls == ["start", "serve", "stop"]
    assert result["terminationReason"] == "deadline_elapsed"
    assert result["captureStateBeforeStop"] == "RUNNING"


@pytest.mark.parametrize("reason", ["source_ended", "source://private-credential"])
def test_source_terminal_reason_survives_cleanup_without_exposing_arbitrary_text(tmp_path, reason):
    module = api()
    calls, events = [], []

    class ClearingRuntime(Runtime):
        async def stop(self):
            self.reason = None
            return await super().stop()

    runtime = ClearingRuntime(calls)

    class EndedServer(Server):
        async def serve(self, *, sockets):
            runtime.state, runtime.reason = "STOPPED", reason
            await super().serve(sockets=sockets)

    result = asyncio.run(
        module.serve_host(
            configuration(tmp_path),
            runtime_factory=lambda _: runtime,
            server_factory=lambda _: EndedServer(calls, wait=True),
            emit=events.append,
        )
    )
    assert result["terminationReason"] == "capture_terminal"
    assert result["captureStateBeforeStop"] == "STOPPED"
    assert result["captureReasonBeforeStop"] == (
        "source_ended" if reason == "source_ended" else "unknown"
    )
    assert "private-credential" not in json.dumps(events)
    assert calls.count("start") == calls.count("stop") == 1


def test_operator_stop_is_distinct_from_deadline(tmp_path):
    module = api()
    calls = []
    requested = asyncio.Event()
    requested.set()
    result = asyncio.run(
        module.serve_host(
            configuration(tmp_path),
            runtime_factory=lambda _: Runtime(calls),
            server_factory=lambda _: Server(calls, wait=True),
            stop_requested=requested,
            emit=lambda _: None,
        )
    )
    assert result["terminationReason"] == "operator_stop"


def test_cancelled_host_drains_capture_and_propagates_cancellation(tmp_path):
    module = api()
    calls, events = [], []
    runtime, server = Runtime(calls), Server(calls, wait=True)

    async def scenario():
        task = asyncio.create_task(
            module.serve_host(
                configuration(tmp_path),
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: server,
                emit=events.append,
            )
        )
        while "serve" not in calls:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls.count("stop") == 1
        assert events[-1]["terminationReason"] == "host_cancelled"

    asyncio.run(scenario())


def test_source_failure_ends_host_without_restarting_it(tmp_path):
    module = api()
    calls, events = [], []
    runtime = Runtime(calls)

    class FailedServer(Server):
        async def serve(self, *, sockets):
            runtime.state = "FAILED"
            runtime.reason = "source_failed"
            await super().serve(sockets=sockets)

    with pytest.raises(module.ConfigurationHostError) as error:
        asyncio.run(
            module.serve_host(
                configuration(tmp_path),
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: FailedServer(calls, wait=True),
                emit=events.append,
            )
        )
    assert error.value.code == "CONFIGURATION_HOST_CAPTURE_FAILED"
    assert calls.count("start") == 1 and calls.count("stop") == 1
    assert events[-1]["terminationReason"] == "capture_terminal"
    assert events[-1]["captureStateBeforeStop"] == "FAILED"
    assert events[-1]["captureReasonBeforeStop"] == "source_failed"


def test_repeated_cancellation_during_stop_must_drain_before_closing_socket(tmp_path):
    module = api()
    calls, events = [], []

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class HeldStop(Runtime):
            async def stop(self):
                self.calls.append("stop")
                entered.set()
                await release.wait()
                return True

        task = asyncio.create_task(
            module.serve_host(
                configuration(tmp_path),
                runtime_factory=lambda _: HeldStop(calls),
                server_factory=lambda _: Server(calls),
                emit=events.append,
            )
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Host must keep owning/draining the original stop"
        with socket.socket() as probe, pytest.raises(OSError):
            probe.bind(("127.0.0.1", events[0]["port"]))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls.count("stop") == 1

    asyncio.run(scenario())


def test_cli_boundary_sanitizes_an_unexpected_exception(tmp_path, monkeypatch, capsys):
    module = api()
    configuration(tmp_path)
    monkeypatch.setenv("SMARTSITE_AI_CAPTURE_SOURCE", "0")
    monkeypatch.setenv("SMARTSITE_AI_CAPTURE_TOKEN", "synthetic-capture-service-token-12345")

    async def failed(_config):
        raise RuntimeError("rtsp://private:credential@example.invalid/source")

    monkeypatch.setattr(module, "_command", failed)
    assert module.main(["--manifest", str(tmp_path / "host.json")]) == 1
    output = capsys.readouterr()
    assert "private:credential" not in output.out + output.err
    assert json.loads(output.out)["code"] == "CONFIGURATION_HOST_SERVER_FAILED"


def test_real_runtime_drain_retains_lock_and_native_read_after_server_ends(tmp_path):
    module = api()
    from configuration_capture_fakes import FakeSource, until

    from smartsite_ai.configuration_runtime import ConfigurationCaptureRuntime

    async def scenario():
        cfg = configuration(tmp_path)
        source = FakeSource()
        source.source_id = cfg.stream.stream_id
        runtime = ConfigurationCaptureRuntime(
            stream=cfg.stream,
            source_kind=cfg.source_kind,
            resource_id=cfg.resource_id,
            lock_directory=cfg.lock_directory,
            service_token=cfg.token,
            source_factory=lambda _: source,
            stop_timeout=0.03,
        )
        other = ConfigurationCaptureRuntime(
            stream=cfg.stream,
            source_kind=cfg.source_kind,
            resource_id=cfg.resource_id,
            lock_directory=cfg.lock_directory,
            service_token=cfg.token,
            source_factory=lambda _: source,
            stop_timeout=0.03,
        )
        events = []

        class AfterRead(Server):
            async def serve(self, *, sockets):
                await source.reading.wait()

        task = asyncio.create_task(
            module.serve_host(
                cfg,
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: AfterRead([]),
                emit=events.append,
            )
        )
        await until(lambda: any(e["state"] == "CONFIGURATION_HOST_DRAIN_PENDING" for e in events))
        assert not task.done() and source.closes == 0
        with pytest.raises(RuntimeError, match="ownership"):
            await other.start()
        source.items.put_nowait(None)
        assert (await task)["nativeReleaseProved"]
        assert source.closes == 1
        await other.start()
        assert await other.stop()  # No scheduling opportunity: source is not reopened.

    asyncio.run(scenario())


def test_explicit_operator_stop_requests_server_exit_without_cancelling_capture(tmp_path):
    module = api()
    calls = []

    async def scenario():
        requested = asyncio.Event()
        runtime, server = Runtime(calls), Server(calls, wait=True)
        task = asyncio.create_task(
            module.serve_host(
                configuration(tmp_path, durationSeconds=10),
                runtime_factory=lambda _: runtime,
                server_factory=lambda _: server,
                stop_requested=requested,
                emit=lambda _: None,
            )
        )
        while "serve" not in calls:
            await asyncio.sleep(0)
        requested.set()
        assert (await task)["nativeReleaseProved"]
        assert server.should_exit and calls.count("stop") == 1

    asyncio.run(scenario())
