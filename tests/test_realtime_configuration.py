import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import WebSocketDisconnect

from smartsite_ai import realtime
from smartsite_ai.config import Settings
from smartsite_ai.domain.observations import PpeObservation
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch, NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.realtime import RealtimeStreamGateTracker
from smartsite_ai.realtime_configuration import RealtimeConfigurationState, read_configuration
from smartsite_ai.tracking.iou_tracker import IoUPersonTracker

PPE = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
OTHER = "f81d4fae-7dec-11d0-a765-00a0c91e6bf7"


def wire():
    return {
        "schemaVersion": "1.0.0",
        "configurationVersion": 1,
        "cameraExternalId": "camera-1",
        "regions": [
            {
                "regionId": region,
                "geometryVersion": 1,
                "coordinateSpace": "NORMALIZED_0_1",
                "polygon": {"coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            }
            for region in (PPE, OTHER)
        ],
    }


def configuration(value):
    return CameraRegionConfiguration.from_wire_bytes(json.dumps(value).encode())


def test_reload_is_explicitly_disabled_by_default():
    assert Settings(_env_file=None).realtime_region_configuration_reload is False


def test_selected_geometry_revision_requires_new_temporal_window():
    current = configuration(wire())
    state = RealtimeConfigurationState(current, PPE)
    replacement = wire()
    replacement["configurationVersion"] = 2
    replacement["regions"][0]["geometryVersion"] = 2
    replacement["regions"][0]["polygon"]["coordinates"][0] = [0.1, 0.1]
    assert state.apply(configuration(replacement)) is True
    assert state.configuration == configuration(replacement)


def test_unrelated_geometry_change_preserves_selected_confirmation():
    state = RealtimeConfigurationState(configuration(wire()), PPE)
    replacement = wire()
    replacement["configurationVersion"] = 2
    replacement["regions"][1]["geometryVersion"] = 2
    replacement["regions"][1]["polygon"]["coordinates"][0] = [0.1, 0.1]
    assert state.apply(configuration(replacement)) is False
    assert state.configuration.configuration_version == 2


@pytest.mark.parametrize(
    "mutation",
    ["camera", "rollback", "missing", "same-version", "polygon-no-bump", "geometry-rollback"],
)
def test_invalid_reload_cannot_replace_previous_configuration(mutation):
    original = wire()
    original["configurationVersion"] = 2
    original["regions"][0]["geometryVersion"] = 2
    current = configuration(original)
    state = RealtimeConfigurationState(current, PPE)
    replacement = deepcopy(original)
    replacement["configurationVersion"] = 3
    if mutation == "camera":
        replacement["cameraExternalId"] = "other-camera"
    elif mutation == "rollback":
        replacement["configurationVersion"] = 1
    elif mutation == "missing":
        replacement["regions"] = replacement["regions"][1:]
    elif mutation == "same-version":
        replacement["configurationVersion"] = 2
        replacement["regions"][0]["geometryVersion"] = 3
    elif mutation == "polygon-no-bump":
        replacement["regions"][0]["polygon"]["coordinates"][0] = [0.1, 0.1]
    else:
        replacement["regions"][0]["geometryVersion"] = 1
    with pytest.raises(ValueError):
        state.apply(configuration(replacement))
    assert state.configuration is current


def test_unchanged_reload_preserves_gate():
    state = RealtimeConfigurationState(configuration(wire()), PPE)
    assert state.apply(configuration(wire())) is False


def test_missing_selected_region_is_rejected_at_initialization():
    value = wire()
    value["regions"] = value["regions"][1:]
    with pytest.raises(ValueError):
        RealtimeConfigurationState(configuration(value), PPE)


def test_reader_rejects_oversize_and_malformed_operator_file(tmp_path):
    path = tmp_path / "regions.json"
    path.write_bytes(b" " * 262145)
    with pytest.raises(ValueError):
        read_configuration(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        read_configuration(path)


def test_reset_clears_old_missing_streak_and_confirmed_episode():
    tracker = RealtimeStreamGateTracker()
    session = UUID("00000000-0000-4000-8000-000000000001")
    start = datetime(2026, 10, 7, tzinfo=UTC)
    obs = PpeObservation.model_validate(
        {
            "type": "PPE",
            "trackId": 1,
            "ppeItem": "HARD_HAT",
            "status": "MISSING",
            "regionId": PPE,
            "geometryVersion": 1,
            "confidence": 0.9,
        }
    )

    def update(gate, index):
        return gate.update(
            stream_id="s",
            session_id=session,
            observed_at=start + timedelta(milliseconds=100 * index),
            active_track_ids=(1,),
            observations=(obs,),
        )

    old = tracker.get_gate(session)
    assert not update(old, 1)
    assert not update(old, 2)
    tracker.reset()
    fresh = tracker.get_gate(session)
    assert fresh is not old
    assert not update(fresh, 3)
    assert not update(fresh, 4)
    assert update(fresh, 5)
    tracker.reset()
    assert not tracker.get_gate(session).confirmed_track_items(
        stream_id="s", session_id=session, active_track_ids=(1,)
    )


@pytest.mark.parametrize("invalid", [False, True])
def test_actual_stream_reload_resets_confirmation_or_stops_before_using_invalid_file(
    tmp_path, monkeypatch, invalid
):
    path = tmp_path / "regions.json"
    initial = wire()
    path.write_text(json.dumps(initial), encoding="utf-8")
    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=PPE,
    )
    detector, runner, source, socket = (MagicMock() for _ in range(4))
    source.connect = AsyncMock()
    source.close = AsyncMock()
    counter = 0

    async def read():
        nonlocal counter
        counter += 1
        if counter == 3:
            replacement = deepcopy(initial)
            replacement["configurationVersion"] = 2
            replacement["regions"][0]["geometryVersion"] = 2
            path.write_text("{" if invalid else json.dumps(replacement), encoding="utf-8")
        return FrameEnvelope(
            stream_id="s",
            camera_external_id="camera-1",
            session_id=UUID("00000000-0000-4000-8000-000000000001"),
            captured_at=datetime(2026, 10, 7, tzinfo=UTC) + timedelta(milliseconds=100 * counter),
            sequence_number=counter,
            width=4,
            height=2,
            payload=bytes([0, 0, 255]) * 8,
        )

    async def detect(envelope):
        box = NormalizedBoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9)
        head = NormalizedBoundingBox(x1=0.3, y1=0.1, x2=0.7, y2=0.3)
        return DetectionBatch.from_frame(
            envelope,
            model_artifact_id="fake",
            model_version="test",
            model_sha256="a" * 64,
            detections=(
                NormalizedDetection(
                    class_id=0, class_name="Person", confidence=0.9, bounding_box=box
                ),
                NormalizedDetection(
                    class_id=1, class_name="NO-Hardhat", confidence=0.9, bounding_box=head
                ),
            ),
        )

    source.read_frame = read
    detector.detect = AsyncMock(side_effect=detect)
    socket.accept = AsyncMock()
    socket.close = AsyncMock()
    sent = []
    processed_versions = []
    process = pipeline.observe

    def observed_process(*args, **kwargs):
        processed_versions.append(kwargs["region_configuration"].regions[0].geometry_version)
        return process(*args, **kwargs)

    pipeline.observe = observed_process

    async def send(payload):
        sent.append(payload)
        if payload.get("type") == "frame" and len(sent) == 5:
            raise WebSocketDisconnect()

    socket.send_json = send
    monkeypatch.setattr(
        realtime,
        "_load_realtime_stack",
        lambda settings: (detector, runner, pipeline, configuration(initial)),
    )
    monkeypatch.setattr(realtime, "OpenCvFrameSource", lambda config: source)
    asyncio.run(
        realtime.stream_realtime(
            socket,
            Settings(
                _env_file=None,
                realtime_source="0",
                realtime_ppe_region_id=PPE,
                realtime_region_configuration_path=str(path),
                realtime_region_configuration_reload=True,
            ),
        )
    )
    source.close.assert_awaited_once()
    runner.close.assert_called_once()
    if invalid:
        assert detector.detect.await_count == 2
        assert [item["type"] for item in sent] == ["frame", "frame", "error"]
        assert "regions.json" not in sent[-1]["message"]
    else:
        assert processed_versions == [1, 1, 2, 2, 2]
        assert [item["detections"][0]["confirmedMissingItems"] for item in sent] == [
            [],
            [],
            [],
            [],
            ["HARD_HAT"],
        ]
