import json
from pathlib import Path

import httpx
import pytest
from test_processing_worker import REGION_ID, FakeStream, RecordingPipeline, _configuration, _frame

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker
from smartsite_ai.tracking import IoUPersonTracker

POLICY_REGION = "33333333-3333-4333-8333-333333333333"
LEGACY_ANCHOR = "55555555-5555-4555-8555-555555555555"


def bundle(version: int = 1) -> CameraConfigurationBundle:
    legacy = _configuration().model_dump(mode="json", by_alias=True)
    region = legacy["regions"][0]
    content = {
        **legacy,
        "configurationVersion": version,
        "cameraId": "44444444-4444-4444-8444-444444444444",
        "viewRevision": 1,
        "issuedAt": "2026-10-27T00:00:00.000Z",
        "regions": [
            {**region, "zoneId": REGION_ID, "policyRevision": version, "requiredPpe": []},
            {
                **region,
                "regionId": POLICY_REGION,
                "zoneId": POLICY_REGION,
                "policyRevision": version,
                "requiredPpe": ["HARD_HAT"],
            },
        ],
    }
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode()
    )


class Detector:
    async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
        return DetectionBatch.from_frame(
            frame,
            model_artifact_id="fake",
            model_version="test",
            model_sha256="a" * 64,
            detections=tuple(
                NormalizedDetection(
                    class_id=index,
                    class_name=name,
                    confidence=0.95,
                    bounding_box=NormalizedBoundingBox(
                        x1=0.2, y1=0.1, x2=0.4, y2=0.5, coordinate_space="NORMALIZED_0_1"
                    ),
                )
                for index, name in enumerate(("person", "no-hardhat"))
            ),
        )


@pytest.mark.anyio
async def test_headless_worker_delivers_policy_of_nonselected_region(tmp_path: Path) -> None:
    store = RegionConfigurationStore()
    active = bundle()
    await store.apply_bundle(active)
    assert await store.get_active_configuration("absent") is None
    assert await store.get_active_configuration("CAM-GATE-01") is active
    projection = await store.get("CAM-GATE-01")
    await store.apply_bundle(active)
    assert await store.get("CAM-GATE-01") is projection
    assert await store.get_active_configuration("CAM-GATE-01") is active
    delivered = []

    def handle(request: httpx.Request) -> httpx.Response:
        event = json.loads(request.content)
        delivered.append(event)
        return httpx.Response(
            202, json={"eventId": event["eventId"], "status": "PROCESSED", "alertIds": []}
        )

    outbox = SqliteEventOutbox(tmp_path / "policy.sqlite3")
    async with BackendClient(
        "http://backend:3000", "test", transport=httpx.MockTransport(handle)
    ) as client:
        worker = HeadlessCameraProcessingWorker(
            stream=FakeStream([_frame(0), _frame(1), _frame(2)]),
            detector=Detector(),
            pipeline=Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline(),
                zones=RestrictedZonePipeline(),
                ppe_region_id=LEGACY_ANCHOR,
            ),
            ppe_region_id=LEGACY_ANCHOR,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, client),
        )
        result = await worker.run()
    assert result.frames_processed == 3
    assert len(delivered) == 1
    ppe = [
        observation for observation in delivered[0]["observations"] if observation["type"] == "PPE"
    ]
    assert [
        (observation["regionId"], observation["ppeItem"], observation["status"])
        for observation in ppe
    ] == [(POLICY_REGION, "HARD_HAT", "MISSING")]


@pytest.mark.anyio
async def test_live_policy_only_replacement_during_inference_discards_old_frame(
    tmp_path: Path,
) -> None:
    store = RegionConfigurationStore()
    await store.apply_bundle(bundle())

    class UpdatingDetector(Detector):
        async def detect(self, frame: FrameEnvelope) -> DetectionBatch:
            result = await super().detect(frame)
            await store.apply_bundle(bundle(2))
            return result

    stream = FakeStream([])
    stream.config = stream.config.model_copy(update={"is_live": True})
    pipeline = RecordingPipeline()
    outbox = SqliteEventOutbox(tmp_path / "stale-policy.sqlite3")
    async with BackendClient("http://backend:3000", "test") as client:
        worker = HeadlessCameraProcessingWorker(
            stream=stream,
            detector=UpdatingDetector(),
            pipeline=pipeline,
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, client),
        )
        await worker._process_frame(_frame(0))
    assert pipeline.configuration_versions == []
    assert not worker._last_frame_enqueued
    assert (await store.get_active_configuration("CAM-GATE-01")).configuration_version == 2


@pytest.mark.anyio
async def test_same_version_legacy_upgrade_starts_new_policy_confirmation(tmp_path: Path) -> None:
    content = bundle().model_dump(mode="json", by_alias=True, exclude={"bundle_sha256"})
    content["regions"][0]["requiredPpe"] = ["HARD_HAT"]
    content["regions"][1]["requiredPpe"] = []
    active = CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode()
    )
    store = RegionConfigurationStore()
    await store.apply(active.to_region_configuration())
    delivered = []

    def handle(request: httpx.Request) -> httpx.Response:
        event = json.loads(request.content)
        delivered.append(event)
        return httpx.Response(
            202, json={"eventId": event["eventId"], "status": "PROCESSED", "alertIds": []}
        )

    outbox = SqliteEventOutbox(tmp_path / "upgrade.sqlite3")
    await outbox.initialize()
    async with BackendClient(
        "http://backend:3000", "test", transport=httpx.MockTransport(handle)
    ) as client:
        dispatcher = OutboxDispatcher(outbox, client)
        worker = HeadlessCameraProcessingWorker(
            stream=FakeStream([]),
            detector=Detector(),
            pipeline=Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline(),
                zones=RestrictedZonePipeline(),
                ppe_region_id=REGION_ID,
            ),
            ppe_region_id=REGION_ID,
            configurations=store,
            outbox=outbox,
            dispatcher=dispatcher,
        )
        await worker._process_frame(_frame(0))
        await worker._process_frame(_frame(1))
        await store.apply_bundle(active)
        await worker._process_frame(_frame(2))
        await dispatcher.drain_once()
        assert not any(obs["type"] == "PPE" for event in delivered for obs in event["observations"])
        await worker._process_frame(_frame(3))
        await worker._process_frame(_frame(4))
        await dispatcher.drain_once()
    ppe_events = [
        event for event in delivered if any(obs["type"] == "PPE" for obs in event["observations"])
    ]
    assert len(ppe_events) == 1
    assert ppe_events[0]["capturedAt"] == _frame(4).captured_at.isoformat()
