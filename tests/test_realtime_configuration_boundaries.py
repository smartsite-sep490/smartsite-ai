import json
from copy import deepcopy

import pytest

from smartsite_ai import realtime_configuration
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.realtime_configuration import RealtimeConfigurationState

PPE = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
OTHER = "f81d4fae-7dec-11d0-a765-00a0c91e6bf7"
THIRD = "f81d4fae-7dec-11d0-a765-00a0c91e6bf8"
FOURTH = "f81d4fae-7dec-11d0-a765-00a0c91e6bf9"


def _wire() -> dict:
    return {
        "schemaVersion": "1.0.0",
        "configurationVersion": 1,
        "cameraExternalId": "camera-1",
        "regions": [
            {
                "regionId": region_id,
                "geometryVersion": 1,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]},
            }
            for region_id in (PPE, OTHER)
        ],
    }


def _config(payload: dict) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.from_wire_bytes(json.dumps(payload).encode("utf-8"))


def test_removed_and_readded_region_cannot_rollback_geometry_or_mutate_polygon() -> None:
    initial_wire = _wire()
    initial_wire["regions"][1]["geometryVersion"] = 2
    v1 = _config(initial_wire)
    state = RealtimeConfigurationState(v1, PPE.upper())

    v2_wire = deepcopy(initial_wire)
    v2_wire["configurationVersion"] = 2
    v2_wire["regions"] = [v2_wire["regions"][0]]
    v2 = _config(v2_wire)
    assert state.apply(v2) is False
    assert state.configuration is v2

    v3_rollback = deepcopy(initial_wire)
    v3_rollback["configurationVersion"] = 3
    v3_rollback["regions"][1]["regionId"] = OTHER.upper()
    v3_rollback["regions"][1]["geometryVersion"] = 1
    with pytest.raises(ValueError, match="Region geometry version decreased"):
        state.apply(_config(v3_rollback))
    assert state.configuration is v2

    v3_mutated_polygon = deepcopy(initial_wire)
    v3_mutated_polygon["configurationVersion"] = 3
    v3_mutated_polygon["regions"][1]["geometryVersion"] = 2
    v3_mutated_polygon["regions"][1]["polygon"]["coordinates"][0] = [0.2, 0.2]
    with pytest.raises(ValueError, match="Region polygon changed without a new geometry version"):
        state.apply(_config(v3_mutated_polygon))
    assert state.configuration is v2

    v3_valid_readd = deepcopy(initial_wire)
    v3_valid_readd["configurationVersion"] = 3
    v3 = _config(v3_valid_readd)
    assert state.apply(v3) is False
    assert state.configuration is v3


def test_retained_region_history_cap_rejects_overflow_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(realtime_configuration, "MAX_RETAINED_REGION_HISTORY", 3)
    initial_wire = _wire()
    initial_wire["regions"][1]["geometryVersion"] = 2
    v1 = _config(initial_wire)
    state = RealtimeConfigurationState(v1, PPE)

    v2_wire = deepcopy(initial_wire)
    v2_wire["configurationVersion"] = 2
    v2_wire["regions"][1]["regionId"] = THIRD
    v2 = _config(v2_wire)
    assert state.apply(v2) is False
    assert state.configuration is v2

    v3_overflow = deepcopy(v2_wire)
    v3_overflow["configurationVersion"] = 3
    v3_overflow["regions"][1]["regionId"] = FOURTH
    with pytest.raises(ValueError, match="Retained region history exceeds the session limit"):
        state.apply(_config(v3_overflow))
    assert state.configuration is v2

    v3_rollback = deepcopy(initial_wire)
    v3_rollback["configurationVersion"] = 3
    v3_rollback["regions"][1]["geometryVersion"] = 1
    with pytest.raises(ValueError, match="Region geometry version decreased"):
        state.apply(_config(v3_rollback))
    assert state.configuration is v2
