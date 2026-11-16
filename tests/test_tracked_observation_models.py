import copy
import importlib
import json
from pathlib import Path

import pytest

from smartsite_ai.domain.observations import TechnicalObservationEvent

ROOT = Path(__file__).resolve().parents[1]


def golden():
    return json.loads(
        (ROOT / "contracts/examples/v1.7/one-person-camera-ppe.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def model():
    name = "smartsite_ai.domain.tracked_observations"
    try:
        return importlib.import_module(name).TrackedObservationEvent
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail("Strict tracked event model is not implemented")


def test_canonical_camera_event_retains_original_values_without_legacy_admission(model):
    event = golden()
    before = copy.deepcopy(event)
    decoded = model.model_validate(event)
    assert decoded.to_wire_dict() == event
    assert event == before
    with pytest.raises(ValueError):
        TechnicalObservationEvent.model_validate(event)


@pytest.mark.parametrize(
    "case",
    [
        "no-person-box",
        "orphan-ppe",
        "duplicate-person",
        "mixed-scope",
        "no-claim",
        "duplicate-claim",
        "active-retired",
        "self-predecessor",
        "bool-index",
        "uint64-overflow",
    ],
)
def test_rejects_inconsistent_original_subject_or_partition(model, case):
    event = golden()
    person = next(item for item in event["observations"] if item["type"] == "PERSON")
    ppe = next(item for item in event["observations"] if item["type"] == "PPE")
    claim = event["continuity"]["subjects"][0]
    if case == "no-person-box":
        person.pop("boundingBox")
    elif case == "orphan-ppe":
        ppe["trackId"] = person["trackId"] + 1
    elif case == "duplicate-person":
        event["observations"].append(copy.deepcopy(person))
    elif case == "mixed-scope":
        regional = {key: value for key, value in ppe.items() if key != "scope"}
        regional.update(regionId="11111111-1111-4111-8111-111111111111", geometryVersion=1)
        event["observations"].append(regional)
    elif case == "no-claim":
        event["continuity"]["subjects"] = []
    elif case == "duplicate-claim":
        event["continuity"]["subjects"].append(copy.deepcopy(claim))
    elif case == "active-retired":
        event["continuity"]["retiredSegments"] = [
            {"segmentId": claim["segmentId"], "reason": "LOST_TRACK"}
        ]
    elif case == "self-predecessor":
        claim["link"] = {
            "kind": "CONTINUE",
            "previousEventId": event["eventId"].lower(),
            "previousPersonObservationIndex": 0,
        }
    elif case == "bool-index":
        claim["personObservationIndex"] = True
    else:
        event["framePartition"]["sequenceNumber"] = str(2**64)
    with pytest.raises(ValueError):
        model.model_validate(event)


def test_incomplete_coverage_keeps_observations_without_inventing_subject_claims(model):
    event = golden()
    event["continuity"]["coverage"] = {"status": "INCOMPLETE", "reason": "FRAME_DROPPED"}
    with pytest.raises(ValueError):
        model.model_validate(event)
    event["continuity"]["subjects"] = []
    assert model.model_validate(event).to_wire_dict() == event
    event["observations"] = []
    assert model.model_validate(event).to_wire_dict() == event


def test_versions_preserve_camera_and_zone_origin_boundaries(model):
    event = golden()
    for version in ("1.4.0", "1.5.0", "1.6.0", "1.8.0"):
        with pytest.raises(ValueError):
            model.model_validate({**event, "schemaVersion": version})
    event["observations"] = [item for item in event["observations"] if item["type"] == "PERSON"]
    assert model.model_validate({**event, "schemaVersion": "1.5.0"})
    entry = {
        "type": "ZONE_ENTRY",
        "trackId": event["observations"][0]["trackId"],
        "regionId": "11111111-1111-4111-8111-111111111111",
        "geometryVersion": 1,
    }
    event["observations"].append(entry)
    assert model.model_validate({**event, "schemaVersion": "1.5.0"})
    with pytest.raises(ValueError):
        model.model_validate({**event, "schemaVersion": "1.6.0"})
    entry["entryOrigin"] = "INSIDE_ENTRY_UNOBSERVED"
    assert model.model_validate({**event, "schemaVersion": "1.6.0"})
    entry["entryOrigin"] = "OUTSIDE_TO_INSIDE"
    with pytest.raises(ValueError):
        model.model_validate({**event, "schemaVersion": "1.6.0"})
    event["continuity"]["subjects"][0]["link"] = {
        "kind": "CONTINUE",
        "previousEventId": "22222222-2222-4222-8222-222222222222",
        "previousPersonObservationIndex": 0,
    }
    assert model.model_validate({**event, "schemaVersion": "1.6.0"})
