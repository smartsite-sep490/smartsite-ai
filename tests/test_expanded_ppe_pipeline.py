import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError
from test_ppe_pipeline import batch, detection, region

from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.inference.ppe_profiles import NATIVE_PPE_10_PROFILE
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.tracking import IoUPersonTracker


def test_orchestrator_selects_expanded_wire_explicitly() -> None:
    from test_mf05_mf06_pipeline import PPE_REGION_ID, configuration
    from test_mf05_mf06_pipeline import batch as frame_batch

    from smartsite_ai.pipelines import Mf05Mf06Pipeline, RestrictedZonePipeline

    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE),
        zones=RestrictedZonePipeline(),
        ppe_region_id=PPE_REGION_ID,
        schema_version="1.1.0",
    )
    event = pipeline.process(
        frame_batch(
            1,
            detection("Person", (0.1, 0.1, 0.9, 0.9)),
            detection("NO-Gloves", (0.2, 0.4, 0.3, 0.5)),
        ),
        region_configuration=configuration(),
        event_id="f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    )
    assert event is not None and event.schema_version == "1.1.0"
    assert any(getattr(obs, "ppe_item", None) == "GLOVES" for obs in event.observations)


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0"])
def test_orchestrator_rejects_incompatible_consumer_before_processing(version: str) -> None:
    from smartsite_ai.pipelines import Mf05Mf06Pipeline, RestrictedZonePipeline

    with pytest.raises(ValueError, match="consumer|unsupported observation"):
        Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE),
            zones=RestrictedZonePipeline(),
            ppe_region_id="f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            schema_version=version,
        )


@pytest.mark.parametrize("item", ["GLOVES", "BOOTS", "GOGGLES"])
def test_extended_temporal_conflict_breaks_confirmation(item: str) -> None:
    from test_ppe_pipeline import SESSION

    from smartsite_ai.domain.observations import PpeObservation
    from smartsite_ai.pipelines.ppe_temporal import TemporalPpeCandidateGate

    gate = TemporalPpeCandidateGate(confirmation_frames=2)
    start = datetime(2026, 10, 6, tzinfo=UTC)
    missing = PpeObservation.model_validate(envelope(item)["observations"][0])
    present = missing.model_copy(update={"status": "PRESENT"})
    for index, observations in enumerate([(missing,), (missing, present), (missing,), (missing,)]):
        confirmed = gate.update(
            stream_id="test",
            session_id=SESSION,
            active_track_ids=(1,),
            observed_at=start + timedelta(milliseconds=100 * index),
            observations=observations,
        )
        assert len(confirmed) == (1 if index == 3 else 0)
    assert confirmed[0].ppe_item == item


def envelope(item: str, version: str = "1.1.0") -> dict:
    return {
        "eventId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
        "schemaVersion": version,
        "cameraExternalId": "CAM-01",
        "streamSessionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf7",
        "capturedAt": "2026-10-06T00:00:00Z",
        "frameDimensions": {"width": 640, "height": 480},
        "observations": [
            {
                "type": "PPE",
                "trackId": 1,
                "ppeItem": item,
                "status": "MISSING",
                "regionId": "f81d4fae-7dec-11d0-a765-00a0c91e6bf8",
                "geometryVersion": 1,
            }
        ],
        "evidence": [],
    }


@pytest.mark.parametrize("item", ["GLOVES", "BOOTS", "GOGGLES"])
def test_expanded_item_requires_explicit_wire_version(item: str) -> None:
    assert TechnicalObservationEvent.model_validate(envelope(item)).schema_version == "1.1.0"
    with pytest.raises(ValidationError):
        TechnicalObservationEvent.model_validate(envelope(item, "1.0.0"))


@pytest.mark.parametrize(
    "positive,negative,item",
    [
        ("Gloves", "NO-Gloves", "GLOVES"),
        ("Boots", "NO-Boots", "BOOTS"),
        ("Goggles", "NO-Goggles", "GOGGLES"),
    ],
)
def test_profile_associates_new_evidence_and_keeps_conflict_unknown(
    positive: str,
    negative: str,
    item: str,
) -> None:
    pipeline = PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE)
    person = detection("Person", (0.1, 0.1, 0.9, 0.9))
    bounds = (0.25, 0.2, 0.4, 0.3)
    for label, status in [(positive, "PRESENT"), (negative, "MISSING")]:
        tracked = IoUPersonTracker().update(batch(person, detection(label, bounds)))
        observed = pipeline.process(tracked, region())
        assert [(o.ppe_item, o.status) for o in observed] == [(item, status)]
    tracked = IoUPersonTracker().update(
        batch(
            person,
            detection(positive, bounds),
            detection(negative, bounds),
        )
    )
    assert pipeline.process(tracked, region()) == ()


def test_native_profile_absence_and_no_vest_never_become_missing() -> None:
    pipeline = PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE)
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.1, 0.1, 0.9, 0.9)),
            detection("NO-Safety Vest", (0.2, 0.4, 0.7, 0.7)),
        )
    )
    assert pipeline.process(tracked, region()) == ()


def test_shared_expanded_evidence_is_not_assigned_to_either_person() -> None:
    pipeline = PpePipeline.for_model_profile(NATIVE_PPE_10_PROFILE)
    tracked = IoUPersonTracker().update(
        batch(
            detection("Person", (0.1, 0.1, 0.7, 0.9)),
            detection("Person", (0.3, 0.1, 0.9, 0.9)),
            detection("NO-Gloves", (0.4, 0.4, 0.5, 0.5)),
        )
    )
    assert pipeline.process(tracked, region()) == ()


def test_vendored_expanded_schema_hash_and_python_agree() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "contracts/metadata.json").read_text(encoding="utf-8-sig"))
    raw = (root / "contracts/schemas/v1.1/technical-observation-event.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == metadata["expandedObservationEvent"]["schemaSha256"]
    assert (
        metadata["expandedObservationEvent"]["sourceCommitSha"]
        == "12afa18817590bd90de088c4c53f6fba7d6c4128"
    )
    validator = jsonschema.Draft202012Validator(
        json.loads(raw), format_checker=jsonschema.FormatChecker()
    )
    for item in ["HARD_HAT", "SAFETY_VEST", "GLOVES", "BOOTS", "GOGGLES"]:
        wire = envelope(item)
        validator.validate(wire)
        assert TechnicalObservationEvent.model_validate(wire).to_wire_dict() == wire
