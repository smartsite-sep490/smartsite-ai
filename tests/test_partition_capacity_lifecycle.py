import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import jsonschema
import pytest
from test_observation_capacity import crowd_frame, observation_counts
from test_processing_worker import FakeStream, _frame
from test_processing_worker_bundle_policy import LEGACY_ANCHOR, Detector, bundle

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.integrations.outbox import OutboxConflict, SqliteEventOutbox
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker
from smartsite_ai.tracking import IoUPersonTracker


def test_partitioned_capacity_matches_vendored_main_schema_and_retains_every_finding():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "contracts/metadata.json").read_text())
    for entry in metadata["technicalObservationVersions"]:
        assert entry["sourceCommitSha"] == "dd71e4c923b496eb1ebe4b06189ff71c1a428f68"
        assert entry["sourcePath"] == entry["path"]
        assert (
            hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["schemaSha256"]
        )
    schema = json.loads(
        (root / "contracts/schemas/v1.4/technical-observation-event.json").read_text()
    )
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    full = crowd_frame()
    parts = full.to_partitioned_events(sequence_number=18446744073709551615)
    assert [len(part.observations) for part in parts] == [252, 28]
    assert observation_counts(o for part in parts for o in part.observations) == observation_counts(
        full.observations
    )
    for part in parts:
        assert list(validator.iter_errors(part.to_wire_dict())) == []
        assert part.frame_partition.sequence_number == "18446744073709551615"
    wire = parts[0].to_wire_dict()
    wire["framePartition"]["sequenceNumber"] = 18446744073709551615
    assert list(validator.iter_errors(wire))


def test_atomic_enqueue_rolls_back_all_new_parts_on_conflicting_sibling(tmp_path):
    async def scenario():
        parts = crowd_frame().to_partitioned_events(sequence_number=3)
        outbox = SqliteEventOutbox(tmp_path / "atomic.sqlite3")
        await outbox.enqueue(parts[1])
        conflicting = parts[1].model_copy(update={"captured_at": "2026-10-30T00:00:00Z"})
        with pytest.raises(OutboxConflict):
            await outbox.enqueue_many((parts[0], conflicting))
        assert (await outbox.counts()).pending == 1
        assert [row.event.event_id for row in await outbox.ready()] == [parts[1].event_id]
        await outbox.enqueue_many(parts)
        await outbox.enqueue_many(parts)
        assert (await outbox.counts()).pending == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["legacy-null-partition", "legacy-null-hash"])
def test_explicit_null_does_not_bypass_version_field_boundaries(case):
    event = crowd_frame().to_events()[0].to_wire_dict()
    if case == "legacy-null-partition":
        event["framePartition"] = None
    else:
        event["evidence"] = [{"kind": "FRAME", "uri": "local://synthetic", "sha256": None}]
    with pytest.raises(ValueError):
        TechnicalObservationEvent.model_validate(event)


def test_cancellation_waits_for_actual_sqlite_commit_and_keeps_zone_state(tmp_path):
    async def scenario():
        store = RegionConfigurationStore()
        await store.apply_bundle(bundle())
        outbox = SqliteEventOutbox(tmp_path / "cancel.sqlite3")
        entered, release = asyncio.Event(), asyncio.Event()

        class DelayedCommit:
            async def enqueue(self, event):
                entered.set()
                await release.wait()
                return await outbox.enqueue(event)

        pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline(),
            zones=RestrictedZonePipeline(entry_confirmation_frames=1),
            ppe_region_id=LEGACY_ANCHOR,
        )
        worker = HeadlessCameraProcessingWorker(
            stream=FakeStream([]),
            detector=Detector(),
            pipeline=pipeline,
            ppe_region_id=LEGACY_ANCHOR,
            configurations=store,
            outbox=DelayedCommit(),
            dispatcher=SimpleNamespace(),
        )
        task = asyncio.create_task(worker._process_frame(_frame(0)))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker._durably_committed
        assert (await outbox.counts()).pending == 1
        await worker._process_frame(_frame(1))
        assert (await outbox.counts()).pending == 1
        with sqlite3.connect(outbox.path) as connection:
            assert connection.execute("SELECT count(*) FROM observation_outbox").fetchone()[0] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("item", ["GLOVES", "BOOTS", "GOGGLES"])
def test_temporal_gate_preserves_explicit_canonical_five_ppe_observations(item):
    gate = TemporalPpeCandidateGate()
    observation = PpeObservation(
        type="PPE",
        trackId=1,
        ppeItem=item,
        status="MISSING",
        regionId=str(UUID(int=1)),
        geometryVersion=1,
    )
    for sequence in range(3):
        candidates = gate.update(
            stream_id="synthetic",
            session_id=UUID(int=2),
            observed_at=datetime(2026, 10, 30, tzinfo=UTC) + timedelta(milliseconds=sequence * 100),
            active_track_ids=(1,),
            observations=(observation,),
        )
    assert len(candidates) == 1
    assert candidates[0].ppe_item == item
