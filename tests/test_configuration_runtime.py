"""Explicit model-free runtime wiring; synthetic sources and local ownership files only."""

import asyncio
import importlib
import importlib.util
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from smartsite_ai.ingestion.config import StreamConfig
from tests.configuration_capture_fakes import NOW, FakeSource, frame, until

RESOURCE = UUID("e98f6475-0660-4c71-a466-4c93d1739ee0")
TOKEN = SecretStr("synthetic-configuration-runtime-service-token")
PATH = "/v1/realtime/configuration-image?cameraExternalId=camera-a"


def runtime_pair(directory: Path, source=None, *, source_factory=None, **changes):
    name = "smartsite_ai.configuration_runtime"
    assert importlib.util.find_spec(name) is not None, "Explicit configuration runtime is missing"
    module = importlib.import_module(name)
    source = source or FakeSource()
    builds = []

    def factory(config):
        builds.append(config)
        return source

    stream = StreamConfig(
        stream_id="configuration-stream", camera_external_id="camera-a", source_url="test://camera"
    )
    runtime = module.ConfigurationCaptureRuntime(
        stream=stream,
        source_kind="LIVE",
        resource_id=RESOURCE,
        lock_directory=directory,
        service_token=TOKEN,
        source_factory=source_factory or factory,
        clock=lambda: NOW,
        stop_timeout=0.03,
        **changes,
    )
    return runtime, source, builds


async def finish(runtime, source):
    source.items.put_nowait(None)
    assert await runtime.stop()


def test_app_construction_and_lifespan_never_build_or_start_a_source(tmp_path):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path)
        async with (
            runtime.app.router.lifespan_context(runtime.app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(runtime.app), base_url="http://test"
            ) as client,
        ):
            response = await client.get(
                PATH, headers={"Authorization": f"Bearer {TOKEN.get_secret_value()}"}
            )
            assert response.status_code == 503
        assert source.connects == 0 and builds == []
        assert list(tmp_path.iterdir()) == []
        assert await runtime.stop()

    asyncio.run(scenario())


def test_explicit_runtime_start_connects_once_and_serves_original_source_frame(tmp_path):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path)
        try:
            await asyncio.gather(runtime.start(), runtime.start())
            await source.reading.wait()
            source.items.put_nowait(frame())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(runtime.app), base_url="http://test"
            ) as client:

                async def ready():
                    async with asyncio.timeout(2):
                        while True:
                            value = await client.get(
                                PATH,
                                headers={"Authorization": f"Bearer {TOKEN.get_secret_value()}"},
                            )
                            if value.status_code == 200:
                                return value
                            await asyncio.sleep(0.001)

                response = await ready()
                assert response.json()["purpose"] == "REGION_CONFIGURATION"
                assert response.json()["frame"]["capturedAt"] == NOW.isoformat()
            assert len(builds) == source.connects == 1
        finally:
            await finish(runtime, source)
        with pytest.raises(RuntimeError):
            await runtime.start()

    asyncio.run(scenario())


def test_competing_runtime_never_builds_source_while_old_native_read_or_close_is_pending(tmp_path):
    async def scenario():
        first, source, _ = runtime_pair(tmp_path)
        second, other, builds = runtime_pair(tmp_path)
        await first.start()
        await source.reading.wait()
        try:
            with pytest.raises(RuntimeError, match="ownership"):
                await second.start()
            assert not await first.stop()
            with pytest.raises(RuntimeError, match="ownership"):
                await second.start()
            assert builds == [] and other.connects == source.closes == 0
            source.close_gate.clear()
            source.items.put_nowait(None)
            await until(lambda: source.closes == 1)
            assert not await first.stop()
            with pytest.raises(RuntimeError, match="ownership"):
                await second.start()
            source.close_gate.set()
            assert await first.stop()
            await second.start()
            await other.reading.wait()
            assert len(builds) == other.connects == 1
        finally:
            source.close_gate.set()
            await finish(first, source)
            await finish(second, other)

    asyncio.run(scenario())


def test_failed_source_release_keeps_process_guard_held_until_explicit_recovery(tmp_path):
    async def scenario():
        runtime, source, _ = runtime_pair(tmp_path)
        other, _, builds = runtime_pair(tmp_path)
        source.close_error = True
        await runtime.start()
        await source.reading.wait()
        source.items.put_nowait(None)
        await until(lambda: runtime.state == "FAILED")
        assert not await runtime.stop()
        with pytest.raises(RuntimeError, match="ownership"):
            await other.start()
        assert builds == []
        runtime._process_lease.release()  # fake failed-close fixture teardown only

    asyncio.run(scenario())


def test_factory_failure_is_closed_and_guard_can_be_released_without_native_io(tmp_path):
    async def scenario():
        # Source creation occurs only after the process guard has been acquired.
        def broken(_):
            raise RuntimeError("private source configuration")

        runtime, source, _ = runtime_pair(tmp_path, source_factory=broken)
        await runtime.start()
        await until(lambda: runtime.state == "FAILED")
        assert source.connects == source.closes == 0
        assert await runtime.stop()
        other, next_source, _ = runtime_pair(tmp_path)
        await other.start()
        await next_source.reading.wait()
        await finish(other, next_source)

    asyncio.run(scenario())


def test_stop_before_first_worker_step_never_constructs_or_connects_source(tmp_path):
    async def scenario():
        runtime, source, builds = runtime_pair(tmp_path)
        await runtime.start()
        assert await runtime.stop()
        assert builds == []
        assert source.connects == source.closes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("preconnected", [False, True])
def test_factory_cannot_adopt_or_close_a_mismatched_or_preconnected_source(tmp_path, preconnected):
    async def scenario():
        source = FakeSource()
        if preconnected:
            source.is_connected = True
        else:
            source.source_id = "another-stream"
        runtime, _, _ = runtime_pair(tmp_path, source)
        await runtime.start()
        await until(lambda: runtime.state == "FAILED")
        assert source.connects == source.closes == 0
        assert await runtime.stop()
        # This runtime never owned the source; it must not close another owner's handle.
        if preconnected:
            assert source.is_connected

    asyncio.run(scenario())


def test_cancelled_stop_waiter_cannot_release_the_process_lock(tmp_path):
    async def scenario():
        runtime, source, _ = runtime_pair(tmp_path)
        other, _, builds = runtime_pair(tmp_path)
        await runtime.start()
        await source.reading.wait()
        waiting = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        try:
            with pytest.raises(RuntimeError, match="ownership"):
                await other.start()
            assert builds == [] and source.closes == 0
        finally:
            await finish(runtime, source)

    asyncio.run(scenario())


def test_replay_requires_pacing_and_preserves_declared_kind_without_model_import(tmp_path):
    async def scenario():
        module = importlib.import_module("smartsite_ai.configuration_runtime")
        source = FakeSource()
        stream = StreamConfig(
            stream_id="configuration-stream",
            camera_external_id="camera-a",
            source_url="test://clip",
            is_live=False,
            pace_replay=True,
        )
        runtime = module.ConfigurationCaptureRuntime(
            stream=stream,
            source_kind="REPLAY",
            resource_id=RESOURCE,
            lock_directory=tmp_path,
            service_token=TOKEN,
            source_factory=lambda _: source,
            clock=lambda: NOW,
        )
        await runtime.start()
        try:
            await source.reading.wait()
            source.items.put_nowait(frame())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(runtime.app), base_url="http://test"
            ) as client:
                async with asyncio.timeout(2):
                    while True:
                        result = await client.get(
                            PATH, headers={"Authorization": f"Bearer {TOKEN.get_secret_value()}"}
                        )
                        if result.status_code == 200:
                            break
                        await asyncio.sleep(0.001)
                assert result.json()["sourceKind"] == "REPLAY"
        finally:
            await finish(runtime, source)

        for changed, kind in [
            (stream.model_copy(update={"pace_replay": False}), "REPLAY"),
            (stream, "LIVE"),
        ]:
            with pytest.raises(ValueError):
                module.ConfigurationCaptureRuntime(
                    stream=changed,
                    source_kind=kind,
                    resource_id=RESOURCE,
                    lock_directory=tmp_path,
                    service_token=TOKEN,
                )

    asyncio.run(scenario())
