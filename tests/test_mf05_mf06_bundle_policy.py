from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.configuration_bundle import CameraConfigurationBundle
from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.tracking import IoUPersonTracker

SESSION = UUID("00000000-0000-4000-8000-000000000001")
CAMERA_ID = "11111111-aaaa-4111-8111-11111111111a"
PPE_REGION_ID = "00000000-0000-4000-8000-000000000002"
SECOND_REGION_ID = "22222222-bbbb-4222-8222-22222222222b"
THIRD_REGION_ID = "33333333-cccc-4333-8333-33333333333c"
ZONE_ID_1 = "44444444-dddd-4444-8444-44444444444d"
ZONE_ID_2 = "55555555-eeee-4555-8555-55555555555e"
ZONE_ID_3 = "66666666-ffff-4666-8666-66666666666f"
FULL_POLYGON = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
OVERLAP_POLYGON_A = ((0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9))
OVERLAP_POLYGON_B = ((0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.8))


def detection(class_name: str, bounds: tuple[float, float, float, float]) -> NormalizedDetection:
    x1, y1, x2, y2 = bounds
    return NormalizedDetection(
        class_id=1,
        class_name=class_name,
        confidence=0.9,
        bounding_box=NormalizedBoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2, coordinate_space="NORMALIZED_0_1"
        ),
    )


def scene_batch(sequence_number: int = 1, *, include_vest: bool = True) -> DetectionBatch:
    detections = [
        detection("person", (0.35, 0.20, 0.55, 0.60)),
        detection("helmet", (0.40, 0.22, 0.48, 0.30)),
    ]
    if include_vest:
        detections.append(detection("vest", (0.38, 0.32, 0.52, 0.50)))
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
        detections=tuple(detections),
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


def revised_bundle(
    bundle: CameraConfigurationBundle, *, view_revision: int
) -> CameraConfigurationBundle:
    content = bundle.model_dump(mode="json", by_alias=True)
    content.pop("bundleSha256")
    content["configurationVersion"] += 1
    content["viewRevision"] = view_revision
    return CameraConfigurationBundle.from_wire_bytes(
        json.dumps({**content, "bundleSha256": compute_canonical_payload_hash(content)}).encode()
    )


def test_new_view_cannot_complete_old_view_pending_zone_confirmation() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(bundle_region(SECOND_REGION_ID, 1, ZONE_ID_2, []))
    for sequence in (1, 2):
        pipeline.observe(
            scene_batch(sequence), region_configuration=bundle, event_id=str(UUID(int=sequence))
        )
    new_view = revised_bundle(bundle, view_revision=2)
    for sequence in (3, 4, 5):
        frame = pipeline.observe(
            scene_batch(sequence), region_configuration=new_view, event_id=str(UUID(int=sequence))
        )
        assert frame is not None
        entries = [obs for obs in frame.observations if obs.type == "ZONE_ENTRY"]
        assert len(entries) == (1 if sequence == 5 else 0)


def test_new_view_invalidates_confirmed_zone_occupancy() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(bundle_region(SECOND_REGION_ID, 1, ZONE_ID_2, []))
    for sequence in (1, 2, 3):
        pipeline.observe(
            scene_batch(sequence), region_configuration=bundle, event_id=str(UUID(int=sequence))
        )
    assert pipeline.occupied_zone_track_regions(
        stream_id="stream-01", session_id=SESSION, active_track_ids=(1,)
    )
    pipeline.observe(
        scene_batch(4),
        region_configuration=revised_bundle(bundle, view_revision=2),
        event_id=str(UUID(int=4)),
    )
    assert not pipeline.occupied_zone_track_regions(
        stream_id="stream-01", session_id=SESSION, active_track_ids=(1,)
    )


def test_same_view_bundle_refresh_preserves_pending_zone_confirmation() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(bundle_region(SECOND_REGION_ID, 1, ZONE_ID_2, []))
    for sequence in (1, 2):
        pipeline.observe(
            scene_batch(sequence), region_configuration=bundle, event_id=str(UUID(int=sequence))
        )
    frame = pipeline.observe(
        scene_batch(3),
        region_configuration=revised_bundle(bundle, view_revision=1),
        event_id=str(UUID(int=3)),
    )
    assert frame is not None
    assert len([obs for obs in frame.observations if obs.type == "ZONE_ENTRY"]) == 1


def test_rejected_new_view_frame_restores_view_with_zone_confirmation_checkpoint() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(bundle_region(SECOND_REGION_ID, 1, ZONE_ID_2, []))
    for sequence in (1, 2):
        pipeline.observe(
            scene_batch(sequence), region_configuration=bundle, event_id=str(UUID(int=sequence))
        )
    checkpoint = pipeline.checkpoint_temporal_state()
    new_view = revised_bundle(bundle, view_revision=2)
    pipeline.observe(scene_batch(3), region_configuration=new_view, event_id=str(UUID(int=3)))
    pipeline.restore_temporal_state(checkpoint)
    for sequence in (4, 5, 6):
        frame = pipeline.observe(
            scene_batch(sequence), region_configuration=new_view, event_id=str(UUID(int=sequence))
        )
        assert frame is not None
        assert len([obs for obs in frame.observations if obs.type == "ZONE_ENTRY"]) == (
            1 if sequence == 6 else 0
        )


def ppe_tuples(event: TechnicalObservationEvent | None) -> set[tuple[str, int, str, str]]:
    assert event is not None
    return {
        (obs.region_id, obs.geometry_version, obs.ppe_item, obs.status)
        for obs in event.observations
        if isinstance(obs, PpeObservation)
    }


def test_bundle_regions_independently_select_configured_ppe_including_nonselected_region() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(
        bundle_region(PPE_REGION_ID, 1, ZONE_ID_1, ["HARD_HAT"], FULL_POLYGON),
        bundle_region(SECOND_REGION_ID, 3, ZONE_ID_2, ["SAFETY_VEST"], OVERLAP_POLYGON_A),
    )

    event = pipeline.process(
        scene_batch(1),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000010",
    )

    assert ppe_tuples(event) == {
        (PPE_REGION_ID, 1, "HARD_HAT", "PRESENT"),
        (SECOND_REGION_ID, 3, "SAFETY_VEST", "PRESENT"),
    }


def test_empty_required_ppe_disables_only_its_region_without_cancelling_overlapping_region() -> (
    None
):
    pipeline = make_pipeline()
    bundle = make_bundle(
        bundle_region(PPE_REGION_ID, 1, ZONE_ID_1, [], FULL_POLYGON),
        bundle_region(
            SECOND_REGION_ID, 3, ZONE_ID_2, ["HARD_HAT", "SAFETY_VEST"], OVERLAP_POLYGON_A
        ),
        bundle_region(THIRD_REGION_ID, 5, ZONE_ID_3, [], OVERLAP_POLYGON_B),
    )

    event = pipeline.process(
        scene_batch(1),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000011",
    )

    assert ppe_tuples(event) == {
        (SECOND_REGION_ID, 3, "HARD_HAT", "PRESENT"),
        (SECOND_REGION_ID, 3, "SAFETY_VEST", "PRESENT"),
    }


def test_unsupported_goggles_and_absent_ppe_on_legacy_profile_remain_unknown() -> None:
    pipeline = make_pipeline()
    bundle = make_bundle(
        bundle_region(PPE_REGION_ID, 1, ZONE_ID_1, ["HARD_HAT", "GOGGLES"], FULL_POLYGON),
        bundle_region(
            SECOND_REGION_ID, 3, ZONE_ID_2, ["GOGGLES", "SAFETY_VEST"], OVERLAP_POLYGON_A
        ),
    )

    first = pipeline.process(
        scene_batch(1, include_vest=True),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000012",
    )
    second = pipeline.process(
        scene_batch(2, include_vest=False),
        region_configuration=bundle,
        event_id="00000000-0000-4000-8000-000000000013",
    )

    assert ppe_tuples(first) == {
        (PPE_REGION_ID, 1, "HARD_HAT", "PRESENT"),
        (SECOND_REGION_ID, 3, "SAFETY_VEST", "PRESENT"),
    }
    assert ppe_tuples(second) == {(PPE_REGION_ID, 1, "HARD_HAT", "PRESENT")}


def test_geometry_only_legacy_configuration_behavior_remains_unchanged() -> None:
    pipeline = make_pipeline()
    legacy_config = CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "camera-01",
            "regions": (
                {
                    "regionId": PPE_REGION_ID,
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": FULL_POLYGON},
                },
                {
                    "regionId": SECOND_REGION_ID,
                    "geometryVersion": 7,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": OVERLAP_POLYGON_A},
                },
            ),
        }
    )

    first = pipeline.process(
        scene_batch(1),
        region_configuration=legacy_config,
        event_id="00000000-0000-4000-8000-000000000014",
    )
    assert ppe_tuples(first) == {
        (PPE_REGION_ID, 1, "HARD_HAT", "PRESENT"),
        (PPE_REGION_ID, 1, "SAFETY_VEST", "PRESENT"),
    }
