import json

import pytest
from test_mf05_mf06_bundle_policy import PPE_REGION_ID, scene_batch
from test_policy_configuration_bundle import content, packet

from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.pipelines.tracked_frame_encoder import encode_complete_tracked_frame
from smartsite_ai.tracking import IoUPersonTracker


def policy_bundle(mode="INHERIT", *, outside=False):
    value = content()
    value["regions"] = [value["regions"][0]]
    value["regions"][0]["regionId"] = PPE_REGION_ID
    value["regions"][0]["polygon"]["coordinates"] = (
        [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] if outside else [[0, 0], [1, 0], [1, 1], [0, 1]]
    )
    value["regions"][0]["ppePolicy"] = {
        "INHERIT": {"mode": "INHERIT"},
        "DISABLED": {"mode": "DISABLED", "reason": "Approved demo exception"},
        "EXPLICIT": {"mode": "EXPLICIT", "requiredPpe": ["SAFETY_VEST"]},
    }[mode]
    return CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))


def pipeline(*, enabled=True):
    return Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=PPE_REGION_ID,
        tracked_v17=enabled,
    )


@pytest.mark.parametrize("mode,count", [("INHERIT", 2), ("EXPLICIT", 1), ("DISABLED", 0)])
def test_inside_region_policy_is_applied_without_camera_fallback(mode, count):
    bundle = policy_bundle(mode)
    frame = pipeline().observe_frame(
        scene_batch(), region_configuration=bundle, event_id="00000000-0000-4000-8000-000000000055"
    )
    findings = [item for item in frame.observations if item.type == "PPE"]
    assert len(findings) == count
    assert all(item.region_id == PPE_REGION_ID for item in findings)
    if mode == "EXPLICIT":
        assert [item.ppe_item for item in findings] == ["SAFETY_VEST"]
    assert frame.camera_configuration_version == bundle.configuration_version
    parts, _ = encode_complete_tracked_frame(frame, scope_sha256="a" * 64, profile_sha256="b" * 64)
    assert parts[0]["schemaVersion"] == "1.7.0"


def test_outside_all_polygons_uses_camera_policy_with_same_person_box():
    bundle = policy_bundle("DISABLED", outside=True)
    frame = pipeline().observe_frame(
        scene_batch(), region_configuration=bundle, event_id="00000000-0000-4000-8000-000000000055"
    )
    findings = [item for item in frame.observations if item.type == "PPE"]
    assert len(findings) == 2
    assert all(item.scope == "CAMERA" for item in findings)
    assert len([item for item in frame.observations if item.type == "PERSON"]) == 1
    parts, _ = encode_complete_tracked_frame(frame, scope_sha256="a" * 64, profile_sha256="b" * 64)
    assert all(
        item.get("scope") == "CAMERA" for item in parts[0]["observations"] if item["type"] == "PPE"
    )
    assert json.dumps(parts)


def test_policy_v2_cannot_silently_drop_camera_rules_in_a_legacy_pipeline():
    with pytest.raises(ValueError, match="v1.7"):
        pipeline(enabled=False).observe_frame(
            scene_batch(),
            region_configuration=policy_bundle(),
            event_id="00000000-0000-4000-8000-000000000055",
        )


@pytest.mark.parametrize("change", ["policy", "view", "profile", "dimensions", "session"])
def test_one_tracked_pipeline_never_continues_or_retires_into_a_different_scope(change):
    from uuid import UUID

    bundle = policy_bundle(outside=True)
    producer = pipeline()
    first = producer.observe_frame(
        scene_batch(1), region_configuration=bundle, event_id="00000000-0000-4000-8000-000000000055"
    )
    batch = scene_batch(2)
    if change in {"policy", "view"}:
        value = bundle.model_dump(mode="json", by_alias=True)
        value.pop("bundleSha256")
        value["configurationVersion"] += 1
        if change == "view":
            value["viewRevision"] += 1
        else:
            value["cameraPpePolicy"] = {"mode": "DISABLED", "reason": "Approved later policy"}
        bundle = CameraPolicyConfigurationBundle.from_wire_bytes(packet(value))
    elif change == "profile":
        batch = batch.model_copy(update={"model_sha256": "c" * 64})
    elif change == "dimensions":
        batch = batch.model_copy(update={"frame_width": 1280})
    else:
        batch = batch.model_copy(
            update={"session_id": UUID("00000000-0000-4000-8000-000000000099")}
        )
    with pytest.raises(RuntimeError, match="scope transition") as error:
        producer.observe_frame(
            batch, region_configuration=bundle, event_id="00000000-0000-4000-8000-000000000056"
        )
    assert type(error.value).__name__ == "TrackedScopeTransitionRequired"
    assert first.subject_segments[0].previous_sequence_number is None
