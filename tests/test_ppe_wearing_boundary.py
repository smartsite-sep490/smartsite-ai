"""AI-A03 synthetic regression tests for object-versus-wearing boundary in PpePipeline.

Uses synthetic fixtures only (no human GT, model weights, or camera runtime).
Reuses real PpePipeline, existing helpers from tests/test_ppe_pipeline.py, and profiles.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_ppe_pipeline import SESSION, batch, detection, overlapping_frame, region

from smartsite_ai.inference.ppe_profiles import (
    LEGACY_PPE_5_PROFILE,
    NATIVE_PPE_10_PROFILE,
    UNIFIED_PPE_15_PROFILE,
)
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate
from smartsite_ai.tracking import IoUPersonTracker

# Synthetic upright full-body person box: height = 0.80 (y1=0.10 .. y2=0.90).
UNIQUE_PERSON_BOUNDS: tuple[float, float, float, float] = (0.20, 0.10, 0.50, 0.90)

# Wrong-position synthetic PPE boxes inside UNIQUE_PERSON_BOUNDS:
# (1) Goggles hanging at chest: center_y = 0.46 -> relative_y = (0.46 - 0.10) / 0.80 = 0.45.
CHEST_HANGING_GOGGLES_BOUNDS: tuple[float, float, float, float] = (0.28, 0.42, 0.42, 0.50)
# (2) Helmet carried near hip: center_y = 0.58 -> relative_y = (0.58 - 0.10) / 0.80 = 0.60.
HIP_CARRIED_HELMET_BOUNDS: tuple[float, float, float, float] = (0.22, 0.52, 0.34, 0.64)

# Correct upper-body synthetic positive control boxes inside UNIQUE_PERSON_BOUNDS:
# Head-worn hardhat: center_y = 0.165 -> relative_y = 0.08125.
HEAD_WORN_HELMET_BOUNDS: tuple[float, float, float, float] = (0.28, 0.11, 0.42, 0.22)
# Eye-level worn goggles: center_y = 0.185 -> relative_y = 0.10625.
EYE_WORN_GOGGLES_BOUNDS: tuple[float, float, float, float] = (0.29, 0.15, 0.41, 0.22)
# Torso-worn safety vest: center_y = 0.43 -> relative_y = 0.4125.
TORSO_WORN_VEST_BOUNDS: tuple[float, float, float, float] = (0.24, 0.28, 0.46, 0.58)


@pytest.mark.parametrize("profile", [NATIVE_PPE_10_PROFILE, UNIFIED_PPE_15_PROFILE])
def test_unique_person_goggles_hanging_at_chest_does_not_yield_present_or_missing(
    profile,
) -> None:
    """(1) Goggles hanging at chest on a unique person must not emit PRESENT or MISSING."""
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS, 0.95),
            detection("Goggles", CHEST_HANGING_GOGGLES_BOUNDS, 0.90),
        )
    )
    pipeline = PpePipeline.for_model_profile(profile, head_item_max_relative_center_y=1 / 3)
    observations = pipeline.process(tracked, region())
    assert all(obs.ppe_item != "GOGGLES" for obs in observations)

    fallback_pipeline = PpePipeline(
        class_names_by_item={"GOGGLES": frozenset({"goggles"})},
        missing_class_names_by_item={"GOGGLES": frozenset({"no-goggles"})},
        emit_missing=True,
        head_item_max_relative_center_y=1 / 3,
    )
    fallback_observations = fallback_pipeline.process(tracked, region())
    assert all(obs.ppe_item != "GOGGLES" for obs in fallback_observations)


@pytest.mark.parametrize(
    "pipeline",
    [
        PpePipeline(head_item_max_relative_center_y=1 / 3),
        PpePipeline(emit_missing=True, head_item_max_relative_center_y=1 / 3),
        PpePipeline.for_model_profile(LEGACY_PPE_5_PROFILE, head_item_max_relative_center_y=1 / 3),
        PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE, head_item_max_relative_center_y=1 / 3),
        PpePipeline.for_model_profile(
            UNIFIED_PPE_15_PROFILE, head_item_max_relative_center_y=1 / 3
        ),
    ],
)
def test_unique_person_helmet_carried_near_hip_does_not_yield_present_or_missing(
    pipeline: PpePipeline,
) -> None:
    """(2) Helmet carried near hip on a unique person must not emit PRESENT or MISSING."""
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS, 0.95),
            detection("Hardhat", HIP_CARRIED_HELMET_BOUNDS, 0.92),
        )
    )
    observations = pipeline.process(tracked, region())
    assert all(obs.ppe_item != "HARD_HAT" for obs in observations)


@pytest.mark.parametrize("profile", [NATIVE_PPE_10_PROFILE, UNIFIED_PPE_15_PROFILE])
def test_unique_person_correct_upper_body_positive_control_emits_present(profile) -> None:
    """(3) Properly positioned head/eye/torso PPE on a unique person emits PRESENT."""
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS, 0.95),
            detection("Hardhat", HEAD_WORN_HELMET_BOUNDS, 0.94),
            detection("Goggles", EYE_WORN_GOGGLES_BOUNDS, 0.91),
            detection("Safety Vest", TORSO_WORN_VEST_BOUNDS, 0.93),
        )
    )
    pipeline = PpePipeline.for_model_profile(profile, head_item_max_relative_center_y=1 / 3)
    observations = pipeline.process(tracked, region())
    assert [(obs.track_id, obs.ppe_item, obs.status) for obs in observations] == [
        (1, "HARD_HAT", "PRESENT"),
        (1, "SAFETY_VEST", "PRESENT"),
        (1, "GOGGLES", "PRESENT"),
    ]


@pytest.mark.parametrize("reverse_people", [False, True])
@pytest.mark.parametrize("swap_track_ids", [False, True])
@pytest.mark.parametrize("emit_missing", [False, True])
def test_overlapping_people_remain_ambiguous_for_upper_body_and_carried_items(
    reverse_people: bool,
    swap_track_ids: bool,
    emit_missing: bool,
) -> None:
    """(4) Overlapping people sharing upper-body or carried PPE boxes remain ambiguous."""
    frame = overlapping_frame(
        detection("Hardhat", (0.45, 0.12, 0.55, 0.22), 0.95),
        detection("Goggles", (0.45, 0.15, 0.55, 0.22), 0.92),
        detection("Safety Vest", (0.45, 0.30, 0.55, 0.58), 0.90),
        detection("Hardhat", (0.45, 0.52, 0.55, 0.64), 0.92),
        detection("Goggles", (0.45, 0.42, 0.55, 0.50), 0.92),
        reverse_people=reverse_people,
        swap_track_ids=swap_track_ids,
    )
    for pipeline in (
        PpePipeline(emit_missing=emit_missing, head_item_max_relative_center_y=1 / 3),
        PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE, head_item_max_relative_center_y=1 / 3),
        PpePipeline.for_model_profile(
            UNIFIED_PPE_15_PROFILE, head_item_max_relative_center_y=1 / 3
        ),
        PpePipeline(
            class_names_by_item={"GOGGLES": {"goggles"}},
            missing_class_names_by_item={},
            emit_missing=emit_missing,
            head_item_max_relative_center_y=1 / 3,
        ),
    ):
        assert pipeline.process(frame, region()) == ()


@pytest.mark.parametrize("label", ["Goggles", "NO-Goggles"])
@pytest.mark.parametrize("reverse", [False, True])
def test_carried_spare_preserves_valid_head_evidence_and_explicit_conflicts(label, reverse) -> None:
    evidence = [
        detection("Goggles", EYE_WORN_GOGGLES_BOUNDS),
        detection(label, CHEST_HANGING_GOGGLES_BOUNDS),
        detection("Safety Vest", TORSO_WORN_VEST_BOUNDS),
    ]
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS),
            *(reversed(evidence) if reverse else evidence),
        )
    )
    observed = PpePipeline.for_model_profile(
        UNIFIED_PPE_15_PROFILE, head_item_max_relative_center_y=1 / 3
    ).process(tracked, region())
    expected = [("SAFETY_VEST", "PRESENT")]
    if label == "Goggles":
        expected.append(("GOGGLES", "PRESENT"))
    assert [(o.ppe_item, o.status) for o in observed] == expected


def test_carried_helmet_cannot_clear_a_confirmed_missing_episode() -> None:
    pipeline = PpePipeline(head_item_max_relative_center_y=1 / 3)
    gate = TemporalPpeCandidateGate(confirmation_frames=2, clear_frames=2)
    person = detection("Person", UNIQUE_PERSON_BOUNDS)
    start = batch(person).captured_at
    for index, helmet in enumerate(
        [
            detection("NO-Hardhat", HEAD_WORN_HELMET_BOUNDS),
            detection("NO-Hardhat", HEAD_WORN_HELMET_BOUNDS),
            detection("Hardhat", HIP_CARRIED_HELMET_BOUNDS),
            detection("Hardhat", HIP_CARRIED_HELMET_BOUNDS),
        ]
    ):
        tracked = IoUPersonTracker().update(batch(person, helmet))
        gate.update(
            stream_id="stream-01",
            session_id=SESSION,
            active_track_ids=(1,),
            observed_at=start + timedelta(seconds=index),
            observations=pipeline.process(tracked, region()),
        )
    assert gate.confirmed_track_items(
        stream_id="stream-01", session_id=SESSION, active_track_ids=(1,)
    ) == frozenset({(1, "HARD_HAT")})


def test_experimental_head_filter_is_disabled_by_default() -> None:
    """Preserve current runtime; legacy object association is not wearing qualification."""
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS),
            detection("Hardhat", HIP_CARRIED_HELMET_BOUNDS),
        )
    )
    for pipeline in (PpePipeline(), PpePipeline.for_model_profile(LEGACY_PPE_5_PROFILE)):
        assert [(o.ppe_item, o.status) for o in pipeline.process(tracked, region())] == [
            ("HARD_HAT", "PRESENT")
        ]


def test_head_filter_keeps_explicit_missing_on_lowered_head() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", UNIQUE_PERSON_BOUNDS),
            detection("NO-Hardhat", HIP_CARRIED_HELMET_BOUNDS),
        )
    )
    observed = PpePipeline(head_item_max_relative_center_y=1 / 3).process(tracked, region())
    assert [(o.ppe_item, o.status) for o in observed] == [("HARD_HAT", "MISSING")]


def test_head_filter_abstains_for_cropped_person_without_creating_missing() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.20, 0.10, 0.50, 1.0)),
            detection("Hardhat", HEAD_WORN_HELMET_BOUNDS),
        )
    )
    observed = PpePipeline(emit_missing=True, head_item_max_relative_center_y=1 / 3).process(
        tracked, region()
    )
    assert all(o.ppe_item != "HARD_HAT" for o in observed)


def test_rejected_positive_cannot_turn_contradictory_raw_evidence_into_missing() -> None:
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.20, 0.10, 0.50, 1.0)),
            detection("Hardhat", HEAD_WORN_HELMET_BOUNDS),
            detection("NO-Hardhat", HEAD_WORN_HELMET_BOUNDS),
        )
    )
    observed = PpePipeline(head_item_max_relative_center_y=1 / 3).process(tracked, region())
    assert observed == ()


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), 0, 1.1, "0.3"])
def test_head_filter_rejects_invalid_limits(invalid) -> None:
    with pytest.raises(ValueError, match="head_item_max_relative_center_y"):
        PpePipeline(head_item_max_relative_center_y=invalid)
