import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import test_processing_worker as fixtures
from test_processing_confirmation_recovery import MissingPpePipeline

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker


def policy_bundle(version, ppe):
    content = fixtures._configuration(1).model_dump(mode="json", by_alias=True)
    content.update(
        cameraId=str(UUID(int=1)),
        configurationVersion=version,
        viewRevision=1,
        issuedAt="2026-10-27T00:00:00.000Z",
    )
    for region in content["regions"]:
        region.update(zoneId=str(UUID(int=3)), policyRevision=version, requiredPpe=[ppe])
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps(
            {
                **content,
                "bundleSha256": compute_canonical_payload_hash(content),
            }
        ).encode()
    )


def test_ppe_confirmation_cannot_span_a_policy_only_configuration_change():
    async def scenario():
        store = RegionConfigurationStore()
        await store.apply_bundle(policy_bundle(1, "HARD_HAT"))
        enqueue = AsyncMock()
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
            detector=fixtures.FakeDetector(),
            pipeline=MissingPpePipeline(),
            ppe_region_id=fixtures.REGION_ID,
            configurations=store,
            outbox=SimpleNamespace(enqueue=enqueue),
            dispatcher=SimpleNamespace(),
        )
        for sequence in range(2):
            await worker._process_frame(fixtures._frame(sequence))
        await store.apply_bundle(policy_bundle(2, "GOGGLES"))
        await worker._process_frame(fixtures._frame(2))
        enqueue.assert_not_awaited()
        for sequence in range(3, 5):
            await worker._process_frame(fixtures._frame(sequence))
        assert enqueue.await_count == 1

    asyncio.run(scenario())
