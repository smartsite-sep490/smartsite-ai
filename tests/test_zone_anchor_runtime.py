from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from test_processing_worker import FakeStream
from test_source_control_child_profile import inputs

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import (
    DetectionBatch,
    NormalizedBoundingBox,
    NormalizedDetection,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.configuration_poller import CameraConfigurationPoller
from smartsite_ai.integrations.outbox import OutboxDispatcher, SqliteEventOutbox
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.processing_worker import HeadlessCameraProcessingWorker
from smartsite_ai.runtime.analysis_preview import encode_analysis_frame
from smartsite_ai.runtime.assembly import CameraPlan, assemble_sessions
from smartsite_ai.runtime.generation_manifest import bind_generation_manifest
from smartsite_ai.runtime.manifest import CameraRuntimeError, bind_manifest, parse_manifest
from smartsite_ai.runtime.source_control_child import bind_source_control_child_memory
from smartsite_ai.source_control_models import SourcePolicy
from smartsite_ai.tracking import IoUPersonTracker
from smartsite_ai.tracking.continuity import FrameContinuityPolicy
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson

CAMERA_ID = UUID("11111111-1111-4111-8111-111111111111")
PPE_REGION_ID = "22222222-2222-4222-8222-222222222222"
ZONE_REGION_ID = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
SESSION_ID = UUID("00000000-0000-4000-8000-000000000001")
BASE_TIME = datetime(2026, 11, 23, 4, 25, 0, tzinfo=UTC)

# Torso/center (0.60, 0.57) inside polygon [0.40..0.85, 0.30..0.85];
# bottom center (0.60, 0.99) outside below polygon bottom edge y=0.85.
CENTER_IN_FOOT_OUT = (0.40, 0.15, 0.80, 0.99)
# Bottom center (0.60, 0.50) inside polygon; box center (0.60, 0.25) above y=0.30.
FOOT_IN_CENTER_OUT = (0.45, 0.00, 0.75, 0.50)
# Shifted outside (center_x=0.39 < 0.40) while preserving IoU > 0.30 with CENTER_IN_FOOT_OUT.
COMPLETELY_OUTSIDE = (0.18, 0.15, 0.60, 0.99)


def _person(bounds: tuple[float, float, float, float]) -> NormalizedDetection:
    return NormalizedDetection(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bounding_box=NormalizedBoundingBox(
            x1=bounds[0],
            y1=bounds[1],
            x2=bounds[2],
            y2=bounds[3],
            coordinate_space="NORMALIZED_0_1",
        ),
    )


def _frame(seq: int, *, camera_external_id: str = "CAM-01") -> FrameEnvelope:
    return FrameEnvelope(
        stream_id="cam-stream-01",
        session_id=SESSION_ID,
        camera_external_id=camera_external_id,
        captured_at=BASE_TIME + timedelta(seconds=seq),
        width=640,
        height=480,
        sequence_number=seq,
        payload=b"\x00\x00\x00" * (640 * 480),
        pixel_format="BGR24",
    )


def _batch(
    seq: int,
    *detections: NormalizedDetection,
    camera_external_id: str = "CAM-01",
) -> DetectionBatch:
    return DetectionBatch.from_frame(
        _frame(seq, camera_external_id=camera_external_id),
        model_artifact_id="test-person-anchored",
        model_version="1",
        model_sha256="a" * 64,
        detections=detections,
    )


def _zone_config(
    geometry_version: int = 1,
    *,
    camera_external_id: str = "CAM-01",
) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": geometry_version,
            "cameraExternalId": camera_external_id,
            "regions": (
                {
                    "regionId": PPE_REGION_ID,
                    "geometryVersion": geometry_version,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {"coordinates": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))},
                },
                {
                    "regionId": ZONE_REGION_ID,
                    "geometryVersion": geometry_version,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {
                        "coordinates": (
                            (0.40, 0.30),
                            (0.85, 0.30),
                            (0.85, 0.85),
                            (0.40, 0.85),
                        )
                    },
                },
            ),
        }
    )


def _zone_only_config(
    geometry_version: int = 1,
    *,
    camera_external_id: str = "CAM-01",
) -> CameraRegionConfiguration:
    return CameraRegionConfiguration.model_validate(
        {
            "schemaVersion": "1.0.0",
            "configurationVersion": geometry_version,
            "cameraExternalId": camera_external_id,
            "regions": (
                {
                    "regionId": ZONE_REGION_ID,
                    "geometryVersion": geometry_version,
                    "coordinateSpace": "NORMALIZED_0_1",
                    "polygon": {
                        "coordinates": (
                            (0.40, 0.30),
                            (0.85, 0.30),
                            (0.85, 0.85),
                            (0.40, 0.85),
                        )
                    },
                },
            ),
        }
    )


def _manifest_doc(tmp_path: Path, *, zone_anchor: object = ...) -> dict[str, object]:
    model_spec = tmp_path / "model.json"
    model_spec.write_text("{}", encoding="utf-8")
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"video-bytes")
    camera: dict[str, object] = {
        "streamId": "cam-stream-01",
        "cameraId": str(CAMERA_ID),
        "cameraExternalId": "CAM-01",
        "ppeRegionId": PPE_REGION_ID,
        "source": {"kind": "path", "path": str(video)},
        "live": False,
        "outbox": str(tmp_path / "outbox.sqlite3"),
    }
    if zone_anchor is not ...:
        camera["zoneAnchor"] = zone_anchor
    return {
        "schemaVersion": "1",
        "modelSpec": str(model_spec),
        "cameras": [camera],
    }


def _assemble_worker(tmp_path: Path, *, zone_anchor: object = ...):
    manifest = parse_manifest(_manifest_doc(tmp_path, zone_anchor=zone_anchor))
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
    return manifest, bound, sessions[0].worker


def test_zone_pipeline_box_center_rearms_after_exit_and_reentry() -> None:
    cfg = _zone_only_config()
    tracker = IoUPersonTracker()
    zones = RestrictedZonePipeline(anchor="BOX_CENTER")

    assert zones.process(tracker.update(_batch(1, _person(CENTER_IN_FOOT_OUT))), cfg) == ()
    assert zones.process(tracker.update(_batch(2, _person(CENTER_IN_FOOT_OUT))), cfg) == ()
    entries = zones.process(tracker.update(_batch(3, _person(CENTER_IN_FOOT_OUT))), cfg)
    assert len(entries) == 1
    assert entries[0].track_id == 1
    assert entries[0].region_id == ZONE_REGION_ID
    assert entries[0].geometry_version == 1

    # Frame 4 stays occupied without duplicate ZONE_ENTRY.
    assert zones.process(tracker.update(_batch(4, _person(CENTER_IN_FOOT_OUT))), cfg) == ()
    assert zones.occupied_track_regions(
        stream_id="cam-stream-01",
        session_id=SESSION_ID,
        active_track_ids=(1,),
    ) == frozenset({(1, ZONE_REGION_ID)})

    # Two consecutive completely-outside frames rearms confirmation.
    assert zones.process(tracker.update(_batch(5, _person(COMPLETELY_OUTSIDE))), cfg) == ()
    assert zones.occupied_track_regions(
        stream_id="cam-stream-01",
        session_id=SESSION_ID,
        active_track_ids=(1,),
    ) == frozenset({(1, ZONE_REGION_ID)})
    assert zones.process(tracker.update(_batch(6, _person(COMPLETELY_OUTSIDE))), cfg) == ()
    assert (
        zones.occupied_track_regions(
            stream_id="cam-stream-01",
            session_id=SESSION_ID,
            active_track_ids=(1,),
        )
        == frozenset()
    )

    assert zones.process(tracker.update(_batch(7, _person(CENTER_IN_FOOT_OUT))), cfg) == ()
    assert zones.process(tracker.update(_batch(8, _person(CENTER_IN_FOOT_OUT))), cfg) == ()
    reentry = zones.process(tracker.update(_batch(9, _person(CENTER_IN_FOOT_OUT))), cfg)
    assert len(reentry) == 1


def test_zone_pipeline_inverse_foot_inside_center_outside_respects_anchor() -> None:
    cfg = _zone_only_config()

    box_center_zones = RestrictedZonePipeline(anchor="BOX_CENTER")
    box_tracker = IoUPersonTracker()
    for seq in range(1, 5):
        assert (
            box_center_zones.process(
                box_tracker.update(_batch(seq, _person(FOOT_IN_CENTER_OUT))), cfg
            )
            == ()
        )
    assert (
        box_center_zones.occupied_track_regions(
            stream_id="cam-stream-01",
            session_id=SESSION_ID,
            active_track_ids=(1,),
        )
        == frozenset()
    )

    default_zones = RestrictedZonePipeline()
    default_tracker = IoUPersonTracker()
    assert (
        default_zones.process(default_tracker.update(_batch(1, _person(FOOT_IN_CENTER_OUT))), cfg)
        == ()
    )
    assert (
        default_zones.process(default_tracker.update(_batch(2, _person(FOOT_IN_CENTER_OUT))), cfg)
        == ()
    )
    assert (
        len(
            default_zones.process(
                default_tracker.update(_batch(3, _person(FOOT_IN_CENTER_OUT))), cfg
            )
        )
        == 1
    )


def test_zone_pipeline_box_center_resets_streaks_on_view_continuity_and_track_gaps() -> None:
    cfg = _zone_only_config()
    policy = FrameContinuityPolicy(
        max_source_sequence_gap=2, max_capture_gap=timedelta(milliseconds=1500)
    )
    zones = RestrictedZonePipeline(anchor="BOX_CENTER", continuity_policy=policy)
    person = _person(CENTER_IN_FOOT_OUT)

    def frame(seq: int, track_id: int | None) -> TrackedFrame:
        persons = () if track_id is None else (TrackedPerson(track_id=track_id, detection=person),)
        return TrackedFrame(
            batch=_batch(seq, *(() if track_id is None else (person,))), persons=persons
        )

    # 1. Missing track frame resets 2-frame pending streak.
    assert zones.process(frame(1, 1), cfg, view_revision=1) == ()
    assert zones.process(frame(2, 1), cfg, view_revision=1) == ()
    assert zones.process(frame(3, None), cfg, view_revision=1) == ()
    assert zones.process(frame(4, 1), cfg, view_revision=1) == ()
    assert zones.process(frame(5, 1), cfg, view_revision=1) == ()

    # 2. View revision change resets streak before frame 6 could confirm.
    assert zones.process(frame(6, 1), cfg, view_revision=2) == ()
    assert zones.process(frame(7, 1), cfg, view_revision=2) == ()

    # 3. Track ID switch resets streak cleanly without inheriting prior track counts.
    assert zones.process(frame(8, 2), cfg, view_revision=2) == ()
    assert zones.process(frame(9, 2), cfg, view_revision=2) == ()

    # 4. Tracked continuity gap (>1500ms at seq=13 -> +4s jump) resets streak cleanly.
    assert zones.process(frame(13, 2), cfg, view_revision=2, include_entry_origin=True) == ()
    assert zones.process(frame(14, 2), cfg, view_revision=2, include_entry_origin=True) == ()
    entries = zones.process(frame(15, 2), cfg, view_revision=2, include_entry_origin=True)
    assert len(entries) == 1
    assert entries[0].track_id == 2


@pytest.mark.parametrize(
    "bad_anchor",
    ["", "center", "box_center", "TOP_CENTER", None, 123, True],
)
def test_zone_pipeline_rejects_invalid_anchor(bad_anchor: object) -> None:
    with pytest.raises(ValueError):
        RestrictedZonePipeline(anchor=bad_anchor)  # type: ignore[arg-type]


def test_manifest_zone_anchor_defaults_roundtrips_and_wires_assembly(tmp_path: Path) -> None:
    default_dir = tmp_path / "default"
    default_dir.mkdir()
    default_manifest, default_bound, default_worker = _assemble_worker(default_dir)
    assert default_manifest.cameras[0].zone_anchor == "BOTTOM_CENTER"
    assert default_bound.cameras[0].entry.zone_anchor == "BOTTOM_CENTER"

    cfg = _zone_config()
    default_pipeline = default_worker._pipeline  # type: ignore[attr-defined]
    for seq in range(1, 5):
        event = default_pipeline.process(
            _batch(seq, _person(CENTER_IN_FOOT_OUT)),
            region_configuration=cfg,
            event_id=f"00000000-0000-4000-8000-{seq:012d}",
        )
        assert event is not None
        assert [o.type for o in event.observations if o.type == "ZONE_ENTRY"] == []

    opt_dir = tmp_path / "opt_in"
    opt_dir.mkdir()
    opt_manifest, opt_bound, opt_worker = _assemble_worker(opt_dir, zone_anchor="BOX_CENTER")
    assert opt_manifest.cameras[0].zone_anchor == "BOX_CENTER"
    assert opt_bound.cameras[0].entry.zone_anchor == "BOX_CENTER"

    opt_pipeline = opt_worker._pipeline  # type: ignore[attr-defined]
    for seq in (1, 2):
        ev = opt_pipeline.process(
            _batch(seq, _person(CENTER_IN_FOOT_OUT)),
            region_configuration=cfg,
            event_id=f"00000000-0000-4000-8000-{seq:012d}",
        )
        assert ev is not None
        assert [o.type for o in ev.observations if o.type == "ZONE_ENTRY"] == []
    confirmed_ev = opt_pipeline.process(
        _batch(3, _person(CENTER_IN_FOOT_OUT)),
        region_configuration=cfg,
        event_id="00000000-0000-4000-8000-000000000003",
    )
    assert confirmed_ev is not None
    assert len([o for o in confirmed_ev.observations if o.type == "ZONE_ENTRY"]) == 1

    # Generation & source-operator binding roundtrip preserves BOX_CENTER.
    roundtrip_dir = tmp_path / "roundtrip"
    roundtrip_dir.mkdir()
    value, bundle, cred_path, _, expected, env = inputs(roundtrip_dir)
    value["cameras"][0]["zoneAnchor"] = "BOX_CENTER"
    parsed = parse_manifest(value)
    gen_bound = bind_generation_manifest(
        parsed, cred_path, UUID(bundle.camera_id), Path(parsed.cameras[0].outbox)
    )
    op_bound, _ = bind_source_control_child_memory(
        gen_bound,
        policy=SourcePolicy({"usb0": 0}, {}, ()),
        scopes={(value["cameras"][0]["trackedV17"]["siteId"], bundle.camera_external_id)},
        camera_id=UUID(bundle.camera_id),
        environ=env,
        expected=expected,
    )
    assert op_bound.cameras[0].zone_anchor == "BOX_CENTER"
    assert bind_manifest(op_bound, env).cameras[0].entry.zone_anchor == "BOX_CENTER"


@pytest.mark.parametrize(
    "bad_anchor",
    ["center", "box_center", "", None, True, 1],
)
def test_manifest_zone_anchor_rejects_invalid_enum(tmp_path: Path, bad_anchor: object) -> None:
    with pytest.raises(CameraRuntimeError):
        parse_manifest(_manifest_doc(tmp_path, zone_anchor=bad_anchor))


def test_analysis_frame_preview_and_worker_emit_zone_anchor_only_when_box_center(
    tmp_path: Path,
) -> None:
    cfg = _zone_config(camera_external_id="CAM-GATE-01")
    frame = _frame(1, camera_external_id="CAM-GATE-01")
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at_iso,
        frame_dimensions={"width": frame.width, "height": frame.height},
        observations=[
            {
                "type": "PERSON",
                "trackId": 1,
                "confidence": 0.92,
                "boundingBox": {
                    "coordinateSpace": "NORMALIZED_0_1",
                    "x1": CENTER_IN_FOOT_OUT[0],
                    "y1": CENTER_IN_FOOT_OUT[1],
                    "x2": CENTER_IN_FOOT_OUT[2],
                    "y2": CENTER_IN_FOOT_OUT[3],
                },
            }
        ],
    )

    default_item = AnalysisFrame(frame, event, cfg, PPE_REGION_ID, frozenset())
    assert default_item.zone_anchor == "BOTTOM_CENTER"
    explicit_bottom = AnalysisFrame(
        frame, event, cfg, PPE_REGION_ID, frozenset(), zone_anchor="BOTTOM_CENTER"
    )
    box_center_item = AnalysisFrame(
        frame, event, cfg, PPE_REGION_ID, frozenset(), zone_anchor="BOX_CENTER"
    )

    default_payload = encode_analysis_frame(default_item)
    bottom_payload = encode_analysis_frame(explicit_bottom)
    box_center_payload = encode_analysis_frame(box_center_item)

    assert "zoneAnchor" not in default_payload
    assert default_payload == bottom_payload
    assert box_center_payload["zoneAnchor"] == "BOX_CENTER"
    assert {k: v for k, v in box_center_payload.items() if k != "zoneAnchor"} == default_payload

    for bad_preview_anchor in ("CENTER", "", 123, None):
        with pytest.raises(ValueError, match="anchor"):
            encode_analysis_frame(
                AnalysisFrame(
                    frame,
                    event,
                    cfg,
                    PPE_REGION_ID,
                    frozenset(),
                    zone_anchor=bad_preview_anchor,  # type: ignore[arg-type]
                )
            )

    async def verify_worker_forwarding() -> None:
        class _StaticDetector:
            async def detect(self, src_frame: FrameEnvelope) -> DetectionBatch:
                return DetectionBatch.from_frame(
                    src_frame,
                    model_artifact_id="test",
                    model_version="1",
                    model_sha256="a" * 64,
                    detections=(_person(CENTER_IN_FOOT_OUT),),
                )

        store = RegionConfigurationStore()
        await store.apply(cfg)
        analyzed: list[AnalysisFrame] = []
        stream = FakeStream([_frame(1, camera_external_id="CAM-GATE-01")])
        stream.config = stream.config.model_copy(update={"stream_id": "cam-stream-01"})
        outbox = SqliteEventOutbox((tmp_path / "worker-outbox.sqlite3").resolve())
        pipeline = Mf05Mf06Pipeline(
            tracker=IoUPersonTracker(),
            ppe=PpePipeline(),
            zones=RestrictedZonePipeline(anchor="BOX_CENTER"),
            ppe_region_id=PPE_REGION_ID,
        )
        async with BackendClient(
            "http://backend:3000",
            "token",
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, json={"ok": True})),
        ) as backend:
            worker = HeadlessCameraProcessingWorker(
                stream=stream,  # type: ignore[arg-type]
                detector=_StaticDetector(),
                pipeline=pipeline,
                ppe_region_id=PPE_REGION_ID,
                configurations=store,
                outbox=outbox,
                dispatcher=OutboxDispatcher(outbox, backend),
                delivery_interval_seconds=0.01,
                analysis_observer=analyzed.append,
            )
            await worker.run()

        assert len(analyzed) == 1
        assert analyzed[0].zone_anchor == "BOX_CENTER"
        assert encode_analysis_frame(analyzed[0])["zoneAnchor"] == "BOX_CENTER"

    asyncio.run(verify_worker_forwarding())
