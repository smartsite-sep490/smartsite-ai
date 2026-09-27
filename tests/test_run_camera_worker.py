from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import SecretStr

from smartsite_ai.config import Settings
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.integrations.outbox import OutboxCounts
from smartsite_ai.processing_worker import ProcessingWorkerResult
from smartsite_ai.tools.run_camera_worker import (
    CameraWorkerRunError,
    _absolute_existing_file,
    _absolute_outbox_path,
    _coordinate,
    _require_ppe_region,
    _resolve_source,
)


@dataclass
class CompletingWorker:
    result: ProcessingWorkerResult

    async def run(self) -> ProcessingWorkerResult:
        await asyncio.sleep(0)
        return self.result


class WaitingPoller:
    def __init__(self) -> None:
        self.stopped = False

    async def run(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()
        self.stopped = True


@pytest.mark.anyio
async def test_coordinator_stops_poller_after_finite_video_completes() -> None:
    expected = ProcessingWorkerResult(
        frames_processed=12,
        events_enqueued=2,
        outbox=OutboxCounts(pending=0, delivered=2, terminal=0),
    )
    poller = WaitingPoller()

    result = await _coordinate(CompletingWorker(expected), poller)  # type: ignore[arg-type]

    assert result == expected
    assert poller.stopped is True


@pytest.mark.anyio
async def test_coordinator_cancels_processing_when_poller_fails() -> None:
    cancelled = asyncio.Event()

    class BlockingWorker:
        async def run(self) -> ProcessingWorkerResult:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError("unreachable")

    class FailingPoller:
        async def run(self, stop_event: asyncio.Event) -> None:
            raise RuntimeError("configuration rejected")

    with pytest.raises(RuntimeError, match="configuration rejected"):
        await _coordinate(  # type: ignore[arg-type]
            BlockingWorker(),
            FailingPoller(),  # type: ignore[arg-type]
        )

    assert cancelled.is_set()


def test_cli_paths_must_be_explicit_and_preexisting(tmp_path: Path) -> None:
    regular = (tmp_path / "model.json").resolve()
    regular.write_text("{}", encoding="utf-8")
    assert _absolute_existing_file(regular, "model spec") == regular
    assert _absolute_outbox_path((tmp_path / "events.sqlite3").resolve()).is_absolute()

    with pytest.raises(CameraWorkerRunError, match="must be absolute"):
        _absolute_existing_file(Path("model.json"), "model spec")
    with pytest.raises(CameraWorkerRunError, match="regular non-symlink"):
        _absolute_existing_file((tmp_path / "missing.json").resolve(), "model spec")
    with pytest.raises(CameraWorkerRunError, match="parent directory"):
        _absolute_outbox_path((tmp_path / "missing" / "events.sqlite3").resolve())


def test_source_prefers_cli_and_supports_secret_environment_setting() -> None:
    settings = Settings(worker_source=SecretStr("rtsp://user:pass@camera/stream"), _env_file=None)

    assert _resolve_source("0", settings) == "0"
    assert _resolve_source(None, settings) == "rtsp://user:pass@camera/stream"
    with pytest.raises(CameraWorkerRunError, match="SMARTSITE_AI_WORKER_SOURCE"):
        _resolve_source(None, Settings(_env_file=None))


@pytest.mark.anyio
async def test_required_ppe_region_is_validated_before_runtime_resources() -> None:
    store = RegionConfigurationStore()
    configuration = CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "CAM-GATE-01",
            "regions": (
                {
                    "regionId": "11111111-1111-4111-8111-111111111111",
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))},
                },
            ),
        }
    )
    await store.apply(configuration)

    await _require_ppe_region(
        store,
        "CAM-GATE-01",
        UUID("11111111-1111-4111-8111-111111111111"),
    )
    with pytest.raises(CameraWorkerRunError, match="PPE region is absent"):
        await _require_ppe_region(
            store,
            "CAM-GATE-01",
            UUID("22222222-2222-4222-8222-222222222222"),
        )
