"""Synthetic worker regression: discarded confirmation must not suppress fresh work."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_processing_worker as fixtures

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker


class MissingPpePipeline:
    def checkpoint_temporal_state(self):
        return None

    def restore_temporal_state(self, state):
        assert state is None

    def observe(self, batch, *, region_configuration, event_id):
        return TechnicalObservationEvent.create(
            event_id=event_id,
            camera_external_id=batch.camera_external_id,
            stream_session_id=str(batch.session_id),
            captured_at=batch.captured_at.isoformat(),
            frame_dimensions={"width": batch.frame_width, "height": batch.frame_height},
            observations=[
                {
                    "type": "PERSON",
                    "trackId": 7,
                    "confidence": 0.9,
                    "boundingBox": {
                        "x1": 0.1,
                        "y1": 0.1,
                        "x2": 0.6,
                        "y2": 0.9,
                        "coordinateSpace": "NORMALIZED_0_1",
                    },
                },
                {
                    "type": "PPE",
                    "trackId": 7,
                    "ppeItem": "HARD_HAT",
                    "status": "MISSING",
                    "confidence": 0.9,
                    "regionId": fixtures.REGION_ID,
                    "geometryVersion": 1,
                },
            ],
        )


def test_live_config_swap_after_confirmation_does_not_latch_an_undelivered_ppe_episode():
    async def scenario():
        store = RegionConfigurationStore()
        original = fixtures._configuration(1)
        await store.apply(original)
        payload = original.model_dump(by_alias=True)
        payload["configurationVersion"] = 2
        replacement = CameraRegionConfiguration.model_validate(payload)
        assert replacement.regions[0].geometry_version == original.regions[0].geometry_version

        class Publisher:
            changed = False

            async def publish(self, *, frame, event):
                if not self.changed:
                    self.changed = True
                    await store.apply(replacement)
                return event

        enqueue = AsyncMock()
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
            detector=fixtures.FakeDetector(),
            pipeline=MissingPpePipeline(),
            ppe_region_id=fixtures.REGION_ID,
            configurations=store,
            outbox=SimpleNamespace(enqueue=enqueue),
            dispatcher=SimpleNamespace(),
            evidence_publisher=Publisher(),
        )
        for sequence in range(3):
            await worker._process_frame(fixtures._frame(sequence))
        enqueue.assert_not_awaited()  # The obsolete version must remain rejected.
        for sequence in range(3, 6):
            await worker._process_frame(fixtures._frame(sequence))
        assert enqueue.await_count == 1, "Undelivered confirmation suppresses the fresh episode"
        delivered = enqueue.await_args.args[0]
        assert any(o.type == "PPE" and o.status == "MISSING" for o in delivered.observations)

    asyncio.run(scenario())


def test_discarded_zone_confirmation_recovers_without_repeating_prior_accepted_occupancy():
    from smartsite_ai.inference.models import NormalizedBoundingBox, NormalizedDetection
    from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
    from smartsite_ai.tracking import IoUPersonTracker

    async def scenario():
        zone_id = "33333333-3333-4333-8333-333333333333"
        payload = fixtures._configuration().model_dump(by_alias=True)
        payload["regions"] += (
            {
                "regionId": zone_id,
                "geometryVersion": 1,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
            },
        )
        store = RegionConfigurationStore()
        await store.apply(CameraRegionConfiguration.model_validate(payload))
        replacement = CameraRegionConfiguration.model_validate(
            {**payload, "configurationVersion": 2}
        )

        def person(x):
            return NormalizedDetection(
                class_id=0,
                class_name="Person",
                confidence=0.9,
                bounding_box=NormalizedBoundingBox(
                    x1=x, y1=0.1, x2=x + 0.15, y2=0.8, coordinate_space="NORMALIZED_0_1"
                ),
            )

        class Detector(fixtures.FakeDetector):
            async def detect(self, frame):
                batch = await super().detect(frame)
                detections = (
                    (person(0.15),) if frame.sequence_number < 3 else (person(0.15), person(0.65))
                )
                return batch.model_copy(update={"detections": detections})

        class Publisher:
            async def publish(self, *, frame, event):
                if frame.sequence_number == 5:
                    await store.apply(replacement)
                return event

        pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline(class_names_by_item={}, missing_class_names_by_item={}),
            zones=RestrictedZonePipeline(),
            ppe_region_id=fixtures.REGION_ID,
        )
        enqueue = AsyncMock()
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
            detector=Detector(),
            pipeline=pipeline,
            ppe_region_id=fixtures.REGION_ID,
            configurations=store,
            outbox=SimpleNamespace(enqueue=enqueue),
            dispatcher=SimpleNamespace(),
            evidence_publisher=Publisher(),
        )
        for sequence in range(6):
            await worker._process_frame(fixtures._frame(sequence))
        assert enqueue.await_count == 1
        for sequence in range(6, 9):
            await worker._process_frame(fixtures._frame(sequence))
        assert enqueue.await_count == 2
        entries = [
            [o.track_id for o in call.args[0].observations if o.type == "ZONE_ENTRY"]
            for call in enqueue.await_args_list
        ]
        assert entries == [[1], [2]]

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_at", ["publisher", "enqueue"])
def test_cancellation_rolls_back_only_before_the_durable_enqueue_boundary(cancel_at):
    async def scenario():
        store = RegionConfigurationStore()
        await store.apply(fixtures._configuration())

        def frame(sequence):
            return fixtures._frame(sequence).model_copy(
                update={
                    "captured_at": fixtures._frame(0).captured_at
                    + timedelta(milliseconds=100 * sequence)
                }
            )

        class Publisher:
            cancelled = False

            async def publish(self, *, frame, event):
                if cancel_at == "publisher" and not self.cancelled:
                    self.cancelled = True
                    raise asyncio.CancelledError()
                return event

        enqueue = AsyncMock(
            side_effect=asyncio.CancelledError() if cancel_at == "enqueue" else None
        )
        worker = HeadlessCameraProcessingWorker(
            stream=SimpleNamespace(config=SimpleNamespace(is_live=True)),
            detector=fixtures.FakeDetector(),
            pipeline=MissingPpePipeline(),
            ppe_region_id=fixtures.REGION_ID,
            configurations=store,
            outbox=SimpleNamespace(enqueue=enqueue),
            dispatcher=SimpleNamespace(),
            evidence_publisher=Publisher(),
        )
        for sequence in range(2):
            await worker._process_frame(frame(sequence))
        with pytest.raises(asyncio.CancelledError):
            await worker._process_frame(frame(2))
        if cancel_at == "publisher":
            enqueue.assert_not_awaited()
        await worker._process_frame(frame(3))
        # The two prior accepted observations survive publisher cancellation;
        # an enqueue with an uncertain commit must not cause another confirmation.
        assert enqueue.await_count == 1

    asyncio.run(scenario())
