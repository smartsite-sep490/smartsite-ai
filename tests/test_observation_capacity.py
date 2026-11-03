from collections import Counter
from dataclasses import replace
from uuid import UUID

import pytest
from test_mf05_mf06_bundle_policy import bundle_region, make_bundle
from test_mf05_mf06_pipeline import batch, detection, make_pipeline

from smartsite_ai.domain.observations import TechnicalObservationEvent, ZoneEntryObservation


def crowd_frame():
    detections = []
    for i in range(10):
        x, y = 0.04 + i % 5 * 0.18, 0.05 + i // 5 * 0.45
        detections.extend(
            [
                detection("person", (x, y, x + 0.12, y + 0.35)),
                detection("hardhat", (x + 0.04, y + 0.02, x + 0.08, y + 0.06)),
                detection("safety vest", (x + 0.02, y + 0.11, x + 0.10, y + 0.24)),
            ]
        )
    config = make_bundle(
        *[
            bundle_region(
                str(UUID(int=i + 1, version=4)),
                1,
                str(UUID(int=i + 101, version=4)),
                ["HARD_HAT", "SAFETY_VEST"],
            )
            for i in range(9)
        ]
    )
    pipeline = make_pipeline()
    for sequence in range(1, 4):
        frame = pipeline.observe(
            batch(sequence, *detections),
            region_configuration=config,
            event_id=str(UUID(int=sequence, version=4)),
        )
    assert frame is not None
    return frame


def observation_counts(observations):
    return Counter(o.model_dump_json(by_alias=True, exclude_none=True) for o in observations)


def test_280_observations_pack_losslessly_with_exact_person_context_and_frame():
    frame = crowd_frame()
    assert len(frame.observations) == 280
    events = frame.to_events()
    assert [len(event.observations) for event in events] == [252, 28]
    assert observation_counts(o for e in events for o in e.observations) == observation_counts(
        frame.observations
    )
    assert len({e.event_id for e in events}) == 2
    assert [e.to_wire_dict() for e in events] == [e.to_wire_dict() for e in frame.to_events()]
    for event in events:
        assert event.captured_at == frame.captured_at
        assert event.stream_session_id == frame.stream_session_id
        assert event.camera_external_id == frame.camera_external_id
        assert event.frame_dimensions == frame.frame_dimensions
        assert TechnicalObservationEvent.model_validate(event.to_wire_dict()) == event
        people = {o.track_id for o in event.observations if o.type == "PERSON"}
        assert all(o.track_id in people for o in event.observations)


def test_exact_capacity_preserves_original_order_and_event_id():
    frame = crowd_frame()
    frame = replace(frame, observations=frame.observations[:256])
    (event,) = frame.to_events()
    assert event.event_id == frame.event_id
    assert tuple(event.observations) == frame.observations


def test_subject_over_capacity_keeps_each_region_together_and_repeats_person_context_only():
    frame = crowd_frame()
    person = next(o for o in frame.observations if o.type == "PERSON")
    zones = tuple(
        ZoneEntryObservation(
            type="ZONE_ENTRY",
            trackId=person.track_id,
            regionId=str(UUID(int=i + 200, version=4)),
            geometryVersion=1,
        )
        for i in range(300)
    )
    frame = replace(frame, observations=(person, *zones))
    events = frame.to_events()
    assert [len(e.observations) for e in events] == [256, 46]
    assert all(e.observations[0] == person for e in events)
    assert observation_counts(
        o for e in events for o in e.observations if o.type != "PERSON"
    ) == observation_counts(zones)


def test_oversized_frame_rejects_missing_person_context_instead_of_fabricating_one():
    frame = crowd_frame()
    frame = replace(frame, observations=tuple(o for o in frame.observations if o.type != "PERSON"))
    with pytest.raises(ValueError, match="PERSON"):
        frame.to_events()


def test_preview_uses_whole_frame_people_and_regions_instead_of_one_transport_part():
    from datetime import datetime

    from smartsite_ai.ingestion.envelope import FrameEnvelope
    from smartsite_ai.realtime import build_realtime_preview

    observed = crowd_frame()
    pixels = FrameEnvelope(
        stream_id="stream-01",
        session_id=UUID(observed.stream_session_id),
        camera_external_id=observed.camera_external_id,
        captured_at=datetime.fromisoformat(observed.captured_at),
        sequence_number=3,
        width=640,
        height=480,
        payload=bytes(640 * 480 * 3),
    )
    preview = build_realtime_preview(observed, pixels)
    assert len(preview["detections"]) == 10
    assert len(preview["zoneDetections"]) == 90
    assert {p["trackId"] for p in preview["detections"]} == {
        o.track_id for o in observed.observations if o.type == "PERSON"
    }


@pytest.mark.anyio
@pytest.mark.parametrize("partitioned", [False, True])
async def test_worker_delivers_all_parts_with_real_outbox_and_exact_frame_images(
    tmp_path, partitioned
):
    import json

    import httpx
    from test_processing_worker import FakeDetector, FakeStream, _configuration, _frame

    from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
    from smartsite_ai.domain.observations import FrameDimensions
    from smartsite_ai.evidence.local_publisher import LocalEvidencePublisher
    from smartsite_ai.integrations.backend_client import BackendClient
    from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
    from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker

    full_frame = crowd_frame()

    class FullFramePipeline:
        def checkpoint_temporal_state(self):
            return None

        def restore_temporal_state(self, state):
            assert state is None

        def observe(self, batch, *, event_id, region_configuration):
            return replace(
                full_frame,
                event_id=event_id,
                camera_external_id=batch.camera_external_id,
                stream_session_id=str(batch.session_id),
                captured_at=batch.captured_at.isoformat(),
                frame_dimensions=FrameDimensions(
                    width=batch.frame_width, height=batch.frame_height
                ),
            )

    store = RegionConfigurationStore()
    await store.apply(_configuration())
    outbox = SqliteEventOutbox((tmp_path / "packed.sqlite3").resolve())
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(
            202, json={"eventId": payload["eventId"], "status": "PROCESSED", "alertIds": []}
        )

    async with BackendClient(
        "http://backend:3000", "test", transport=httpx.MockTransport(handler)
    ) as backend:
        worker = HeadlessCameraProcessingWorker(
            stream=FakeStream([_frame(1)]),
            detector=FakeDetector(),
            pipeline=FullFramePipeline(),
            ppe_region_id=_configuration().regions[0].region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, backend),
            evidence_publisher=LocalEvidencePublisher(root_dir=tmp_path, bind_content_hash=True),
            **({"partitioned_frame_delivery": True} if partitioned else {}),
        )
        if not partitioned:
            with pytest.raises(ValueError, match="v1.4"):
                await worker.run()
            assert payloads == []
            assert (await outbox.counts()).pending == 0
            assert list(tmp_path.glob("*.jpg")) == []
            return
        result = await worker.run()
    assert result.frames_processed == 1
    assert result.events_enqueued == result.outbox.delivered == 2
    assert result.outbox.pending == 0
    events = [TechnicalObservationEvent.model_validate(p) for p in payloads]
    assert sorted(len(e.observations) for e in events) == [28, 252]
    assert observation_counts(o for e in events for o in e.observations) == observation_counts(
        full_frame.observations
    )
    assert all(
        e.schema_version == ("1.4.0" if partitioned else "1.2.0") and len(e.evidence) == 1
        for e in events
    )
    if partitioned:
        assert len({e.frame_partition.frame_id for e in events}) == 1
        assert {e.frame_partition.sequence_number for e in events} == {"1"}
        assert {e.frame_partition.part_index for e in events} == {0, 1}
        assert {e.frame_partition.part_count for e in events} == {2}
    else:
        assert all(e.frame_partition is None for e in events)
    assert len({e.evidence[0].uri for e in events}) == 2
    assert len({e.evidence[0].sha256 for e in events}) == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 2
