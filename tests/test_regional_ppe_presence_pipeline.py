"""Supplemental positive evidence reuses region ownership, never required-PPE policy."""

import pytest
from test_mf05_mf06_bundle_policy import detection, scene_batch
from test_tracked_policy_pipeline import policy_bundle

from smartsite_ai.inference.ppe_profiles import NATIVE_PPE_10_PROFILE, UNIFIED_PPE_15_PROFILE
from smartsite_ai.pipelines import Mf05Mf06Pipeline, PpePipeline, RestrictedZonePipeline
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate, tracked_delivery_frame
from smartsite_ai.pipelines.tracked_frame_encoder import encode_complete_tracked_frame
from smartsite_ai.tracking import IoUPersonTracker

EXTRA = {"MASK", "FULL_BODY_HARNESS", "LANYARD_HOOK"}


def observe(
    *,
    mode="INHERIT",
    outside=False,
    unified=True,
    extras=True,
    competing=False,
    negative_mask=False,
):
    batch = scene_batch()
    objects = list(batch.detections)
    if extras:
        for name in ("Mask", "fullbody-Harness", "Lanyard-Hook"):
            objects.append(detection(name, (0.4, 0.32, 0.48, 0.5)))
    if negative_mask:
        objects.append(detection("NO-Mask", (0.4, 0.32, 0.48, 0.5)))
    if competing:
        objects.append(detection("person", (0.36, 0.2, 0.56, 0.6)))
    batch = batch.model_copy(update={"detections": tuple(objects)})
    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline.for_model_profile(UNIFIED_PPE_15_PROFILE) if unified else PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=policy_bundle().regions[0].region_id,
        schema_version="1.1.0",
        tracked_v17=True,
    )
    return batch, pipeline.observe_frame(
        batch,
        region_configuration=policy_bundle(mode, outside=outside),
        event_id="00000000-0000-4000-8000-000000000055",
    )


@pytest.mark.parametrize("mode", ["INHERIT", "EXPLICIT"])
def test_unified_regional_positive_objects_survive_temporal_filter_and_tracked_encoding(mode):
    batch, frame = observe(mode=mode)
    extras = [o for o in frame.observations if o.type == "PPE" and o.ppe_item in EXTRA]
    assert {o.ppe_item for o in extras} == EXTRA
    assert all(o.status == "PRESENT" and o.bounding_box is not None for o in extras)
    # A negative Mask class is not a new required-PPE policy or alert here.
    assert all(o.status != "MISSING" for o in extras)
    gate = TemporalPpeCandidateGate()
    confirmed = gate.update(
        stream_id=batch.stream_id,
        session_id=batch.session_id,
        observed_at=batch.captured_at,
        active_track_ids=[1],
        observations=[o for o in frame.observations if o.type == "PPE"],
        camera_configuration_version=1,
    )
    assert not confirmed
    delivered = tracked_delivery_frame(frame, confirmed)
    parts, _ = encode_complete_tracked_frame(
        delivered, scope_sha256="a" * 64, profile_sha256="b" * 64
    )
    payload = [o for p in parts for o in p["observations"] if o.get("ppeItem") in EXTRA]
    assert len(payload) == 3
    assert all(o["status"] == "PRESENT" and "regionId" in o and "scope" not in o for o in payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "DISABLED"},
        {"outside": True},
        {"unified": False},
        {"extras": False},
        {"competing": True},
    ],
)
def test_no_synthetic_extra_evidence_outside_enabled_unambiguous_regional_scope(kwargs):
    _, frame = observe(**kwargs)
    assert not [o for o in frame.observations if o.type == "PPE" and o.ppe_item in EXTRA]


def test_contradictory_mask_evidence_abstains_without_affecting_harness_hook():
    _, frame = observe(negative_mask=True)
    extras = [o for o in frame.observations if o.type == "PPE" and o.ppe_item in EXTRA]
    assert {o.ppe_item for o in extras} == {"FULL_BODY_HARNESS", "LANYARD_HOOK"}


def test_negative_mask_alone_does_not_create_new_policy_alert():
    _, frame = observe(extras=False, negative_mask=True)
    assert not [o for o in frame.observations if o.type == "PPE" and o.ppe_item in EXTRA]


def test_policy_capability_does_not_silently_claim_extra_required_items():
    for profile in (NATIVE_PPE_10_PROFILE, UNIFIED_PPE_15_PROFILE):
        pipeline = PpePipeline.for_model_profile(profile)
        assert not set(pipeline.supported_items) & EXTRA
        assert pipeline.regional_presence_enabled == (profile == UNIFIED_PPE_15_PROFILE)


def test_exact_analyzed_preview_keeps_existing_five_family_policy_shape():
    from smartsite_ai.ingestion.envelope import FrameEnvelope
    from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
    from smartsite_ai.runtime.analysis_preview import encode_analysis_frame

    batch, frame = observe()
    source = FrameEnvelope(
        stream_id=batch.stream_id,
        session_id=batch.session_id,
        camera_external_id=batch.camera_external_id,
        captured_at=batch.captured_at,
        sequence_number=batch.sequence_number,
        width=batch.frame_width,
        height=batch.frame_height,
        payload=bytes(batch.frame_width * batch.frame_height * 3),
    )
    result = encode_analysis_frame(
        AnalysisFrame(
            source, frame, policy_bundle(), policy_bundle().regions[0].region_id, frozenset()
        )
    )
    assert result["detections"]
    assert all(not set(person["ppeStatus"]) & EXTRA for person in result["detections"])
