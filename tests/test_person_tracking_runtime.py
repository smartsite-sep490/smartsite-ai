from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.runtime.assembly import CameraPlan, assemble_sessions
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest

CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")
REGION_ID = UUID("22222222-2222-4222-8222-222222222222")
SESSION_ID = UUID("00000000-0000-4000-8000-000000000042")
BASE_TIME = datetime(2026, 11, 22, 21, 55, 0, tzinfo=UTC)


def _manifest_doc(
    tmp_path: Path,
    *,
    person_tracking: object = ...,
) -> dict[str, object]:
    model_spec = tmp_path / "model.json"
    model_spec.write_text("{}", encoding="utf-8")
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"video-bytes")
    camera: dict[str, object] = {
        "streamId": "cam-stream-01",
        "cameraId": str(CAMERA_ID),
        "cameraExternalId": "CAM-01",
        "ppeRegionId": str(REGION_ID),
        "source": {"kind": "path", "path": str(video)},
        "live": False,
        "outbox": str(tmp_path / "outbox.sqlite3"),
    }
    if person_tracking is not ...:
        camera["personTracking"] = person_tracking
    return {
        "schemaVersion": "1",
        "modelSpec": str(model_spec),
        "cameras": [camera],
    }


def _assemble_single_camera_worker(tmp_path: Path, *, person_tracking: object = ...):
    from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller

    manifest = parse_manifest(_manifest_doc(tmp_path, person_tracking=person_tracking))
    bound = bind_manifest(manifest, {})
    store = RegionConfigurationStore()
    client = SimpleNamespace()
    poller = CameraConfigurationPoller(
        client=client,  # type: ignore[arg-type]
        store=store,
        camera_id=CAMERA_ID,
        camera_external_id="CAM-01",
    )
    sessions = assemble_sessions(
        [
            CameraPlan(
                bound=bound.cameras[0],
                store=store,
                poller=poller,
                configuration_client=client,  # type: ignore[arg-type]
            )
        ],
        SimpleNamespace(),  # type: ignore[arg-type]
        backend_client_factory=lambda: SimpleNamespace(),  # type: ignore[arg-type,return-value]
        source_factory=lambda cfg: SimpleNamespace(config=cfg),
        evidence_max_bytes=1024,
        delivery_interval_seconds=0.25,
    )
    return manifest, sessions[0].worker


def _region_configuration() -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": 1,
            "cameraExternalId": "CAM-01",
            "regions": (
                {
                    "regionId": str(REGION_ID),
                    "geometryVersion": 1,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))},
                },
            ),
        }
    )


def _person(
    bounds: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=0,
        class_name="person",
        confidence=confidence,
        bounding_box=NormalizedBoundingBox(
            x1=bounds[0],
            y1=bounds[1],
            x2=bounds[2],
            y2=bounds[3],
            coordinate_space="NORMALIZED_0_1",
        ),
    )


def _batch(sequence_number: int, *detections: NormalizedDetection) -> DetectionBatch:
    frame = FrameEnvelope(
        stream_id="cam-stream-01",
        session_id=SESSION_ID,
        camera_external_id="CAM-01",
        captured_at=BASE_TIME + timedelta(seconds=sequence_number),
        width=640,
        height=480,
        sequence_number=sequence_number,
        payload=b"\x00\x00\x00" * (640 * 480),
        pixel_format="BGR24",
    )
    return DetectionBatch.from_frame(
        frame,
        model_artifact_id="test-yolo11s",
        model_version="1",
        model_sha256="a" * 64,
        detections=detections,
    )


def test_manifest_person_tracking_opt_in_wires_tracker_and_preserves_legacy_defaults(
    tmp_path: Path,
) -> None:
    default_dir = tmp_path / "default"
    default_dir.mkdir()
    default_manifest, default_worker = _assemble_single_camera_worker(default_dir)
    assert default_manifest.cameras[0].person_tracking is None
    default_tracker = default_worker._pipeline._tracker  # type: ignore[attr-defined]
    assert default_tracker._new_track_confidence is None
    assert default_tracker._confirmation_frames == 1
    assert default_tracker._association_iou_margin is None

    # Legacy defaults admit even a weak single-frame detection immediately.
    legacy_event = default_worker._pipeline.process(  # type: ignore[attr-defined]
        _batch(1, _person((0.1, 0.1, 0.4, 0.9), confidence=0.28)),
        region_configuration=_region_configuration(),
        event_id="00000000-0000-4000-8000-000000000101",
    )
    assert legacy_event is not None
    assert [o.track_id for o in legacy_event.observations if o.type == "PERSON"] == [1]

    opt_in_dir = tmp_path / "opt_in"
    opt_in_dir.mkdir()
    config = {
        "newTrackConfidence": 0.6,
        "newTrackConfirmationFrames": 2,
        "associationIouMargin": 0.1,
    }
    opt_in_manifest, opt_in_worker = _assemble_single_camera_worker(
        opt_in_dir, person_tracking=config
    )
    assert opt_in_manifest.cameras[0].person_tracking is not None
    tracker = opt_in_worker._pipeline._tracker  # type: ignore[attr-defined]
    assert tracker._new_track_confidence == pytest.approx(0.6)
    assert tracker._confirmation_frames == 2
    assert tracker._association_iou_margin == pytest.approx(0.1)


@pytest.mark.parametrize(
    "bad_tracking",
    [
        {"newTrackConfidence": True, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {"newTrackConfidence": False, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {"newTrackConfidence": 0.0, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {"newTrackConfidence": -0.1, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {"newTrackConfidence": 1.1, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {
            "newTrackConfidence": float("nan"),
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": 0.1,
        },
        {
            "newTrackConfidence": float("inf"),
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": 0.1,
        },
        {"newTrackConfidence": "0.6", "newTrackConfirmationFrames": 2, "associationIouMargin": 0.1},
        {
            "newTrackConfidence": 0.6,
            "newTrackConfirmationFrames": True,
            "associationIouMargin": 0.1,
        },
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 0, "associationIouMargin": 0.1},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 65, "associationIouMargin": 0.1},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2.0, "associationIouMargin": 0.1},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2, "associationIouMargin": False},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2, "associationIouMargin": 0.0},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2, "associationIouMargin": -0.05},
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2, "associationIouMargin": 1.05},
        {
            "newTrackConfidence": 0.6,
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": float("nan"),
        },
        {
            "newTrackConfidence": 0.6,
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": float("-inf"),
        },
        {"newTrackConfidence": 0.6, "newTrackConfirmationFrames": 2, "associationIouMargin": "0.1"},
        {
            "newTrackConfidence": 0.6,
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": 0.1,
            "extraKey": 1,
        },
    ],
)
def test_manifest_person_tracking_rejects_invalid_values(
    tmp_path: Path, bad_tracking: dict[str, object]
) -> None:
    with pytest.raises(CameraRuntimeError):
        parse_manifest(_manifest_doc(tmp_path, person_tracking=bad_tracking))


def test_person_tracking_suppresses_weak_and_single_frame_duplicates_while_retaining_two_people(
    tmp_path: Path,
) -> None:
    _, worker = _assemble_single_camera_worker(
        tmp_path,
        person_tracking={
            "newTrackConfidence": 0.6,
            "newTrackConfirmationFrames": 2,
            "associationIouMargin": 0.1,
        },
    )
    pipeline = worker._pipeline  # type: ignore[attr-defined]
    config = _region_configuration()

    def process(seq: int, *detections: NormalizedDetection) -> list[int]:
        event = pipeline.process(
            _batch(seq, *detections),
            region_configuration=config,
            event_id=f"00000000-0000-4000-8000-{seq:012d}",
        )
        return (
            [] if event is None else [o.track_id for o in event.observations if o.type == "PERSON"]
        )

    # Strong person confirms on frame 2 while overlapping weak false boxes (0.28, 0.55) never admit.
    real = _person((0.0, 0.08, 0.68, 0.99), confidence=0.88)
    weak_curtain = _person((0.0, 0.01, 0.18, 0.98), confidence=0.28)
    weak_split = _person((0.12, 0.05, 0.78, 0.99), confidence=0.55)
    assert process(1, real, weak_curtain) == []
    confirmed_ids = process(2, real, weak_curtain, weak_split)
    assert len(confirmed_ids) == 1
    primary_id = confirmed_ids[0]

    # Single-frame high-confidence non-overlapping phantom (0.76) does not confirm;
    # primary retains its track ID.
    ephemeral_high = _person((0.72, 0.15, 0.98, 0.95), confidence=0.76)
    assert process(3, real, ephemeral_high, weak_split) == [primary_id]
    assert process(4, real, weak_curtain) == [primary_id]

    # Empty frame clears solo primary track before two-person scene.
    assert process(5) == []
    person_a = _person((0.10, 0.10, 0.50, 0.90), confidence=0.87)
    person_b = _person((0.30, 0.10, 0.70, 0.90), confidence=0.84)
    assert process(6, person_a, person_b) == []
    two_ids = process(7, person_a, person_b)
    assert len(two_ids) == 2
    assert len(set(two_ids)) == 2
    # Confidence swaps on frame 8; explicit associationIouMargin preserves both tracks.
    swapped = process(
        8,
        _person((0.10, 0.10, 0.50, 0.90), confidence=0.72),
        _person((0.30, 0.10, 0.70, 0.90), confidence=0.95),
    )
    assert swapped == two_ids

    # Ambiguous equal-overlap crossing still retires both prior tracks permanently.
    crossed = process(9, _person((0.20, 0.10, 0.60, 0.90), confidence=0.90))
    assert crossed == []
    assert set(process(10, person_a, person_b)).isdisjoint(two_ids)
    post_cross = process(11, person_a, person_b)
    assert len(post_cross) == 2
    assert set(post_cross).isdisjoint(two_ids)


def test_binding_roundtrips_preserve_person_tracking(tmp_path: Path) -> None:
    from test_source_control_child_profile import inputs

    from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
    from smartsite_ai.runtime.source_control_child import bind_source_control_child_memory
    from smartsite_ai.source_control_models import SourcePolicy

    value, bundle, credential_path, _, expected, env = inputs(tmp_path)
    value["cameras"][0]["personTracking"] = {
        "newTrackConfidence": 0.6,
        "newTrackConfirmationFrames": 2,
        "associationIouMargin": 0.1,
    }
    manifest = parse_manifest(value)
    gen_bound = bind_generation_manifest(
        manifest,
        credential_path,
        UUID(bundle.camera_id),
        Path(manifest.cameras[0].outbox),
    )
    assert gen_bound.cameras[0].person_tracking == manifest.cameras[0].person_tracking

    bound_manifest, _credential = bind_source_control_child_memory(
        gen_bound,
        policy=SourcePolicy({"usb0": 0}, {}, ()),
        scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )

    assert bound_manifest.cameras[0].source.kind == "env"
    assert bound_manifest.cameras[0].person_tracking == manifest.cameras[0].person_tracking
    runtime_bound = bind_manifest(bound_manifest, env)
    assert runtime_bound.cameras[0].entry.person_tracking is not None
    assert runtime_bound.cameras[0].entry.person_tracking.new_track_confidence == pytest.approx(0.6)
    assert runtime_bound.cameras[0].entry.person_tracking.new_track_confirmation_frames == 2
    assert runtime_bound.cameras[0].entry.person_tracking.association_iou_margin == pytest.approx(
        0.1
    )
