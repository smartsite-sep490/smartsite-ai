"""Private preview hosts reserve ownership before any durable worker starts."""

import asyncio
import socket
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest
from configuration_capture_fakes import FakeSource, until
from pydantic import SecretStr
from test_configuration_capture_host import configuration

from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.ingestion.worker import StreamWorker
from smartsite_ai.integrations.outbox import OutboxCounts
from smartsite_ai.processing_worker import ProcessingWorkerResult
from smartsite_ai.runtime.manifest import CameraRuntimeError


def group(tmp_path, **changes):
    from smartsite_ai.runtime.preview_host import PreviewHostGroup

    options = {"rawPreviewEnabled": True, "durationSeconds": 2, **changes}
    config = configuration(tmp_path, **options)
    bound = SimpleNamespace(
        cameras=(
            SimpleNamespace(
                stream_id=config.stream.stream_id,
                camera_external_id=config.stream.camera_external_id,
                source_config=config.stream,
            ),
        )
    )
    return PreviewHostGroup(bound, (config,)), config


def test_host_port_conflict_releases_reservation_before_worker_construction(tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        hosts, config = group(tmp_path, port=occupied.getsockname()[1])
        with pytest.raises(CameraRuntimeError, match="preview"):
            hosts.reserve()
    lease = ConfigurationProcessLease(config.lock_directory, config.resource_id)
    lease.acquire()
    lease.release()


def test_host_uses_same_physical_lease_as_configuration_capture(tmp_path):
    hosts, config = group(tmp_path)
    lease = ConfigurationProcessLease(config.lock_directory, config.resource_id)
    lease.acquire()
    try:
        with pytest.raises(CameraRuntimeError, match="preview"):
            hosts.reserve()
    finally:
        lease.release()


class Server:
    def __init__(self, app):
        self.app = app
        self.should_exit = False
        self.fail = asyncio.Event()

    async def serve(self, *, sockets):
        assert sockets[0].getsockname()[0] == "127.0.0.1"
        while not self.should_exit:
            if self.fail.is_set():
                raise RuntimeError("synthetic private credential")
            await asyncio.sleep(0.001)


class Worker:
    def __init__(self, stream):
        self.stream = stream
        self.finish = asyncio.Event()

    async def run(self):
        try:
            await self.stream.start()
            await self.finish.wait()
            return ProcessingWorkerResult(0, 0, OutboxCounts(0, 0, 0))
        finally:
            await self.stream.stop()


@pytest.mark.parametrize("ending", ["normal", "server", "cancel", "deadline"])
def test_host_drains_source_before_releasing_physical_lease(tmp_path, ending):
    async def scenario():
        hosts, config = group(tmp_path, durationSeconds=0.1 if ending == "deadline" else 2)
        hosts.reserve()
        source = FakeSource()
        source.source_id = config.stream.stream_id
        stream = StreamWorker(
            config.stream, source, frame_observer=hosts.observer_for(config.stream.stream_id)
        )
        worker = Worker(stream)
        servers = []

        def server_factory(app):
            server = Server(app)
            servers.append(server)
            return server

        wrapped = hosts.wrap(stream, worker, server_factory=server_factory)
        task = asyncio.create_task(wrapped.run())
        competitor = ConfigurationProcessLease(config.lock_directory, config.resource_id)
        try:
            await asyncio.wait_for(source.connected.wait(), 2)
            source.close_gate.clear()
            if ending == "normal":
                worker.finish.set()
            elif ending == "server":
                servers[0].fail.set()
            elif ending == "cancel":
                task.cancel()
            await until(lambda: source.closes == 1)
            assert not task.done()
            with pytest.raises(RuntimeError, match="ownership"):
                competitor.acquire()
            if ending == "cancel":
                task.cancel()  # Repeated cancellation must not abandon owned cleanup.
            source.close_gate.set()
            if ending == "normal":
                assert (await asyncio.wait_for(task, 2)).frames_processed == 0
            elif ending in ("server", "deadline"):
                with pytest.raises(CameraRuntimeError, match="preview"):
                    await asyncio.wait_for(task, 2)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            await hosts.close()
            competitor.acquire()
            competitor.release()
            assert source.connects == source.closes == 1
        finally:
            source.close_gate.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await hosts.close()

    asyncio.run(scenario())


def test_host_rejects_mismatched_camera_source_and_replay(tmp_path):
    from smartsite_ai.runtime.preview_host import PreviewHostGroup

    hosts, config = group(tmp_path)
    for stream in (
        config.stream.model_copy(update={"camera_external_id": "other"}),
        config.stream.model_copy(update={"source_url": SecretStr("other-source")}),
        config.stream.model_copy(update={"is_live": False}),
    ):
        bound = SimpleNamespace(
            cameras=(
                SimpleNamespace(
                    stream_id=config.stream.stream_id,
                    camera_external_id=stream.camera_external_id,
                    source_config=stream,
                ),
            )
        )
        with pytest.raises(CameraRuntimeError, match="preview"):
            PreviewHostGroup(bound, (config,))


def test_failed_native_release_retains_lease_without_retry(tmp_path, monkeypatch):
    from smartsite_ai.runtime import preview_host as module

    releases = []

    class Lease:
        def __init__(self, *_):
            pass

        def acquire(self):
            pass

        def release(self):
            releases.append(True)

    monkeypatch.setattr(module, "ConfigurationProcessLease", Lease)

    async def scenario():
        hosts, config = group(tmp_path)
        hosts.reserve()
        source = FakeSource()
        source.close_error = True
        stream = StreamWorker(
            config.stream, source, frame_observer=hosts.observer_for(config.stream.stream_id)
        )
        worker = Worker(stream)
        task = asyncio.create_task(hosts.wrap(stream, worker, server_factory=Server).run())
        await asyncio.wait_for(source.connected.wait(), 2)
        worker.finish.set()
        with pytest.raises(CameraRuntimeError, match="release is unresolved"):
            await asyncio.wait_for(task, 2)
        with pytest.raises(CameraRuntimeError, match="release is unresolved"):
            await hosts.close()
        assert releases == [] and source.closes == 1
        assert not stream.source_released

    asyncio.run(scenario())


def test_preview_failure_does_not_stop_healthy_sibling_camera(tmp_path):
    from smartsite_ai.runtime.preview_host import PreviewHostGroup
    from smartsite_ai.runtime.supervisor import CameraSession, supervise_sessions

    async def scenario():
        _, first = group(tmp_path)
        second = replace(
            first,
            resource_id=UUID("bce244c6-68a1-4af8-a1f7-c032040301fa"),
            stream=first.stream.model_copy(
                update={
                    "stream_id": "second",
                    "camera_external_id": "SECOND",
                    "source_url": SecretStr("1"),
                }
            ),
        )
        configs = (first, second)
        bound = SimpleNamespace(
            cameras=tuple(
                SimpleNamespace(
                    stream_id=c.stream.stream_id,
                    camera_external_id=c.stream.camera_external_id,
                    source_config=c.stream,
                )
                for c in configs
            )
        )
        hosts = PreviewHostGroup(bound, configs)
        hosts.reserve()
        servers, sources, workers, sessions = [], [], [], []

        class Poller:
            async def run(self, stop):
                await stop.wait()

        def factory(app):
            server = Server(app)
            servers.append(server)
            return server

        for config in configs:
            source = FakeSource()
            source.source_id = config.stream.stream_id
            stream = StreamWorker(
                config.stream, source, frame_observer=hosts.observer_for(config.stream.stream_id)
            )
            worker = Worker(stream)
            sources.append(source)
            workers.append(worker)
            sessions.append(
                CameraSession(
                    config.stream.stream_id,
                    config.stream.camera_external_id,
                    hosts.wrap(stream, worker, server_factory=factory),
                    Poller(),
                )
            )
        task = asyncio.create_task(supervise_sessions(sessions))
        try:
            await until(lambda: len(servers) == 2 and all(s.connects == 1 for s in sources))
            servers[0].fail.set()
            await until(lambda: sources[0].closes == 1)
            assert sources[1].closes == 0 and not task.done()
            workers[1].finish.set()
            outcomes = await asyncio.wait_for(task, 2)
            assert [item.status for item in outcomes] == ["failed", "completed"]
            assert all(s.connects == s.closes == 1 for s in sources)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await hosts.close()

    asyncio.run(scenario())
