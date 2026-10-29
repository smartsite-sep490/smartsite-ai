from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.observations import (
    PpeObservation,
    TechnicalObservationEvent,
    ZoneEntryObservation,
)
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.runtime.assembly import _require_ppe_region as runtime_require_ppe_region
from smartsite_ai.runtime.manifest import CameraRuntimeError
from smartsite_ai.tools.run_camera_worker import (
    CameraWorkerRunError,
)
from smartsite_ai.tools.run_camera_worker import (
    _require_ppe_region as worker_require_ppe_region,
)
from smartsite_ai.tracking import IoUPersonTracker

SESSION = UUID("00000000-0000-4000-8000-000000000001")
CAMERA_ID = "11111111-aaaa-4111-8111-11111111111a"
PPE_REGION_ID = "00000000-0000-4000-8000-000000000002"
SECOND_REGION_ID = "22222222-bbbb-4222-8222-22222222222b"
ZONE_ID_1 = "44444444-dddd-4444-8444-44444444444d"
ZONE_ID_2 = "55555555-eeee-4555-8555-55555555555e"
FULL_POLYGON = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
OVERLAP_POLYGON_A = ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9))


def detection(class_name: str, bounds: tuple[float, float, float, float]) -> NormalizedDetection:
    x1, y1, x2, y2 = bounds
    box = NormalizedBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, coordinate_space="NORMALIZED_0_1")
    return NormalizedDetection(class_id=1, class_name=class_name, confidence=0.9, bounding_box=box)


def scene_batch(sequence_number: int = 1) -> DetectionBatch:
    return DetectionBatch(
        stream_id="stream-01",
        session_id=SESSION,
        camera_external_id="camera-01",
        captured_at=datetime(2026, 10, 27, 12, 0, sequence_number, tzinfo=UTC),
        frame_width=640,
        frame_height=480,
        sequence_number=sequence_number,
        model_artifact_id="fake-detector",
        model_version="test",
        model_sha256="a" * 64,
        detections=(
            detection("person", (0.35, 0.20, 0.55, 0.60)),
            detection("helmet", (0.40, 0.22, 0.48, 0.30)),
            detection("vest", (0.38, 0.32, 0.52, 0.50)),
        ),
    )


def bundle_region(
    region_id: str,
    geometry_version: int,
    zone_id: str,
    required_ppe: list[str],
    coordinates: tuple[tuple[float, float], ...] = FULL_POLYGON,
) -> dict[str, Any]:
    return {
        "regionId": region_id,
        "geometryVersion": geometry_version,
        "coordinateSpace": "NORMALIZED_0_1",
        "polygon": {"coordinates": [list(point) for point in coordinates]},
        "zoneId": zone_id,
        "policyRevision": 1,
        "requiredPpe": required_ppe,
    }


def make_bundle(*regions: dict[str, Any]) -> CameraConfigurationBundle:
    content: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "cameraId": CAMERA_ID,
        "cameraExternalId": "camera-01",
        "configurationVersion": 1,
        "viewRevision": 1,
        "issuedAt": "2026-10-27T08:00:00.000Z",
        "regions": list(regions),
    }
    payload = {**content, "bundleSha256": compute_canonical_payload_hash(content)}
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )


def make_pipeline() -> Mf05Mf06Pipeline:
    return Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=PPE_REGION_ID,
    )


def ppe_tuples(event: TechnicalObservationEvent | None) -> set[tuple[str, int, str, str]]:
    assert event is not None
    return {
        (obs.region_id, obs.geometry_version, obs.ppe_item, obs.status)
        for obs in event.observations
        if isinstance(obs, PpeObservation)
    }


def zone_tuples(event: TechnicalObservationEvent | None) -> set[tuple[str, int]]:
    assert event is not None
    return {
        (obs.region_id, obs.geometry_version)
        for obs in event.observations
        if isinstance(obs, ZoneEntryObservation)
    }


def test_full_bundle_without_legacy_selected_region_still_emits_configured_ppe() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(
        bundle_region(
            SECOND_REGION_ID, 3, ZONE_ID_2, ["HARD_HAT", "SAFETY_VEST"], OVERLAP_POLYGON_A
        )
    )
    event = pipeline.process(
        scene_batch(1),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000020",
    )
    assert ppe_tuples(event) == {
        (SECOND_REGION_ID, 3, "HARD_HAT", "PRESENT"),
        (SECOND_REGION_ID, 3, "SAFETY_VEST", "PRESENT"),
    }


def test_full_bundle_selected_region_yields_zone_entry_after_three_frames_and_ppe() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(bundle_region(PPE_REGION_ID, 2, ZONE_ID_1, ["HARD_HAT"], FULL_POLYGON))
    events = [
        pipeline.process(
            scene_batch(seq),
            region_configuration=bundle,
            event_id=f"00000000-0000-4000-8000-00000000002{seq}",
        )
        for seq in (1, 2, 3)
    ]
    assert zone_tuples(events[0]) == set()
    assert zone_tuples(events[1]) == set()
    assert ppe_tuples(events[2]) == {(PPE_REGION_ID, 2, "HARD_HAT", "PRESENT")}
    assert zone_tuples(events[2]) == {(PPE_REGION_ID, 2)}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("require_ppe_region", "expected_error"),
    [
        (worker_require_ppe_region, CameraWorkerRunError),
        (runtime_require_ppe_region, CameraRuntimeError),
    ],
)
async def test_require_ppe_region_preflight_accepts_full_bundle_and_rejects_missing_legacy(
    require_ppe_region: Any,
    expected_error: type[Exception],
) -> None:
    unmatched_bundle_store = RegionConfigurationStore()
    await unmatched_bundle_store.apply_bundle(
        make_bundle(bundle_region(SECOND_REGION_ID, 3, ZONE_ID_2, ["HARD_HAT"], OVERLAP_POLYGON_A))
    )
    await require_ppe_region(unmatched_bundle_store, "camera-01", UUID(PPE_REGION_ID))

    empty_bundle_store = RegionConfigurationStore()
    await empty_bundle_store.apply_bundle(make_bundle())
    await require_ppe_region(empty_bundle_store, "camera-01", UUID(PPE_REGION_ID))

    geometry_only_store = RegionConfigurationStore()
    await geometry_only_store.apply(
        CameraRegionConfiguration.model_validate(
            {
                "schemaVersion": "1.0.0",
                "configurationVersion": 1,
                "cameraExternalId": "camera-01",
                "regions": (
                    {
                        "regionId": SECOND_REGION_ID,
                        "geometryVersion": 1,
                        "coordinateSpace": "NORMALIZED_0_1",
                        "polygon": {"coordinates": OVERLAP_POLYGON_A},
                    },
                ),
            }
        )
    )
    with pytest.raises(expected_error, match="PPE region is absent"):
        await require_ppe_region(geometry_only_store, "camera-01", UUID(PPE_REGION_ID))
