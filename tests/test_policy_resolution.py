import pytest
from test_policy_configuration_bundle import content, packet

from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle


def bundle(value=None):
    return CameraPolicyConfigurationBundle.from_wire_bytes(packet(value or content()))


def resolve(model, region_ids):
    method = getattr(model, "resolve_ppe_policy", None)
    assert callable(method), "The v2 bundle cannot resolve effective location PPE yet"
    return method(region_ids)


def test_camera_default_applies_outside_polygons_without_creating_a_region():
    value = bundle()
    result = resolve(value, ())
    assert result.scope == "CAMERA"
    assert result.required_ppe == ("HARD_HAT", "SAFETY_VEST")
    assert result.contributions[0].region_id is None
    assert result.contributions[0].source == "CAMERA_DEFAULT"
    assert value.regions[0].ppe_policy.mode == "INHERIT"


def test_explicit_override_and_disabled_binding_do_not_cancel_other_overlaps():
    value = bundle()
    inherit, explicit, disabled = (region.region_id for region in value.regions)
    assert resolve(value, (explicit,)).required_ppe == ("GLOVES",)
    assert resolve(value, (disabled,)).required_ppe == ()
    overlap = resolve(value, (inherit, explicit, disabled))
    assert overlap.scope == "REGIONS"
    assert overlap.required_ppe == ("GLOVES", "HARD_HAT", "SAFETY_VEST")
    assert tuple(entry.region_id for entry in overlap.contributions) == (
        inherit,
        explicit,
        disabled,
    )
    assert tuple(entry.mode for entry in overlap.contributions) == (
        "INHERIT",
        "EXPLICIT",
        "DISABLED",
    )


def test_disabled_camera_default_cannot_cancel_explicit_region():
    data = content()
    data["cameraPpePolicy"] = {"mode": "DISABLED", "reason": "Synthetic exception"}
    value = bundle(data)
    assert resolve(value, ()).required_ppe == ()
    assert resolve(value, tuple(region.region_id for region in value.regions)).required_ppe == (
        "GLOVES",
    )


def test_unknown_duplicate_or_invalid_region_selection_does_not_fall_back():
    value = bundle()
    known = value.regions[0].region_id
    for selected in (("unknown",), (known, known), (None,), known, tuple("x" for _ in range(65))):
        with pytest.raises(ValueError):
            resolve(value, selected)
    result = resolve(value, (known,))
    with pytest.raises(AttributeError):
        result.required_ppe = ()
    assert resolve(value, (known,)).required_ppe == ("HARD_HAT", "SAFETY_VEST")
