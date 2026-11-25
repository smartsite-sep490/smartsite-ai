"""Typed parity with existing immutable regional PPE wire, not model accuracy."""

import copy
import json
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = [json.loads(path.read_text()) for path in (ROOT / "contracts/schemas").rglob("*.json")]
REGISTRY = Registry().with_resources(
    (schema["$id"], Resource.from_contents(schema)) for schema in RESOURCES if "$id" in schema
)


def event_with(item, status="PRESENT", version="1.7.0"):
    event = json.loads((ROOT / "contracts/examples/v1.7/one-person-camera-ppe.json").read_text())
    ppe = next(o for o in event["observations"] if o["type"] == "PPE")
    ppe.pop("scope")
    ppe.update(
        ppeItem=item,
        status=status,
        regionId="11111111-1111-4111-8111-111111111111",
        geometryVersion=1,
    )
    event["schemaVersion"] = version
    return event


@pytest.mark.parametrize("version", ["1.5.0", "1.6.0", "1.7.0"])
@pytest.mark.parametrize(
    "item,status",
    [
        ("MASK", "PRESENT"),
        ("MASK", "MISSING"),
        ("FULL_BODY_HARNESS", "PRESENT"),
        ("LANYARD_HOOK", "PRESENT"),
    ],
)
def test_existing_canonical_regional_records_roundtrip_without_widening_legacy(
    version, item, status
):
    event = event_with(item, status, version)
    schema = next(
        s
        for s in RESOURCES
        if s.get("$id")
        == f"https://smartsite.local/contracts/v{version[:3]}/technical-observation-event.json"
    )
    jsonschema.Draft202012Validator(
        schema, registry=REGISTRY, format_checker=jsonschema.FormatChecker()
    ).validate(event)
    before = copy.deepcopy(event)
    assert TrackedObservationEvent.model_validate(event).to_wire_dict() == before
    assert event == before
    with pytest.raises(ValueError):
        TechnicalObservationEvent.model_validate(event)
    with pytest.raises(ValueError):
        PpeObservation.model_validate(event["observations"][1])


@pytest.mark.parametrize("item", ["FULL_BODY_HARNESS", "LANYARD_HOOK"])
def test_present_only_objects_cannot_become_missing(item):
    with pytest.raises(ValueError):
        TrackedObservationEvent.model_validate(event_with(item, "MISSING"))


@pytest.mark.parametrize("item", ["MASK", "FULL_BODY_HARNESS", "LANYARD_HOOK"])
def test_camera_scope_does_not_silently_gain_new_items(item):
    event = event_with(item)
    ppe = event["observations"][1]
    ppe.pop("regionId")
    ppe.pop("geometryVersion")
    ppe["scope"] = "CAMERA"
    with pytest.raises(ValueError):
        TrackedObservationEvent.model_validate(event)


@pytest.mark.parametrize(
    "change",
    ["missing_region", "bad_region", "bool_geometry", "no_person", "mixed_scope", "extra_field"],
)
def test_additional_regional_ppe_retains_context_and_subject_guards(change):
    event = event_with("FULL_BODY_HARNESS")
    ppe = event["observations"][1]
    if change == "missing_region":
        ppe.pop("regionId")
    elif change == "bad_region":
        ppe["regionId"] = "not-a-region"
    elif change == "bool_geometry":
        ppe["geometryVersion"] = True
    elif change == "no_person":
        ppe["trackId"] += 1
    elif change == "extra_field":
        ppe["permission"] = "allowed"
    else:
        camera_ppe = {k: v for k, v in ppe.items() if k not in {"regionId", "geometryVersion"}}
        camera_ppe.update(ppeItem="HARD_HAT", scope="CAMERA")
        event["observations"].append(camera_ppe)
    with pytest.raises(ValueError):
        TrackedObservationEvent.model_validate(event)
