"""Raw model objects stay separate from wearer observations and business state."""

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from test_multi_camera_runtime import CAMERA_A, _document, _entry, _video
from test_policy_configuration_bundle import content as policy_content
from test_policy_configuration_bundle import packet as policy_packet
from test_processing_worker import FakeStream, _configuration

from smartsite_ai.core.region_configuration_store import RegionConfigurationStore
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.evidence.models import FrameBatchBindingError
from smartsite_ai.inference.models import DetectionBatch, NormalizedDetection
from smartsite_ai.inference.person_anchored import PersonAnchoredPpeDetector
from smartsite_ai.inference.ppe_profiles import LEGACY_PPE_5_PROFILE, UNIFIED_PPE_15_PROFILE
from smartsite_ai.ingestion.testing import make_test_frame
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
from smartsite_ai.runtime.manifest import bind_manifest, parse_manifest
from smartsite_ai.tracking import IoUPersonTracker


def sample(detections=None):
    configuration = _configuration()
    frame = make_test_frame(
        "object-preview", 17, camera_external_id=configuration.camera_external_id
    )
    if detections is None:
        detections = tuple(
            NormalizedDetection(
                class_id=class_id,
                class_name=class_name,
                confidence=0.9,
                bounding_box={"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8},
            )
            for class_id, class_name in UNIFIED_PPE_15_PROFILE.class_map
        )
    batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="synthetic-unified15",
        model_version="test-v1",
        model_sha256="a" * 64,
        detections=detections,
    )
    return frame, configuration, batch


def preview_item(frame, configuration, batch):
    return AnalysisFrame(frame, None, configuration, "ppe", frozenset(), raw_detections=batch)


def test_all_eight_families_survive_as_same_frame_raw_objects_without_wearer_claims():
    frame, configuration, batch = sample()
    payload = encode_analysis_frame(preview_item(frame, configuration, batch))
    objects = payload["modelObjects"]
    assert objects["modelFamily"] == "yolo11s"
    assert objects["experimental"] is True
    assert objects["scope"] == "OBJECT_DETECTION_ONLY"
    assert objects["artifactId"] == batch.model_artifact_id
    assert objects["artifactSha256"] == batch.model_sha256
    assert [(d["classId"], d["className"]) for d in objects["detections"]] == list(
        UNIFIED_PPE_15_PROFILE.class_map
    )
    for item in objects["detections"]:
        assert set(item) == {"classId", "className", "confidence", "boundingBox"}
        assert item["confidence"] == 0.9
        assert item["boundingBox"] == {
            "x1": 0.1,
            "y1": 0.2,
            "x2": 0.6,
            "y2": 0.8,
            "coordinateSpace": "NORMALIZED_0_1",
        }
    assert payload["detections"] == []
    assert payload["zoneDetections"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream_id", "different-stream"),
        ("camera_external_id", "different-camera"),
        ("session_id", UUID("99999999-9999-4999-8999-999999999999")),
        ("sequence_number", 18),
        ("captured_at", "different-time"),
        ("frame_width", 3),
        ("frame_height", 3),
    ],
)
def test_raw_object_batch_cannot_cross_frame_binding(field, value):
    frame, configuration, batch = sample()
    if field == "captured_at":
        value = batch.captured_at + timedelta(seconds=1)
    wrong = batch.model_copy(update={field: value})
    with pytest.raises(FrameBatchBindingError, match="frame"):
        encode_analysis_frame(preview_item(frame, configuration, wrong))


@pytest.mark.parametrize("class_id,class_name", [(11, "NO-Mask"), (4, "Gloves"), (99, "Other")])
def test_raw_object_class_id_and_name_must_match_reviewed_taxonomy(class_id, class_name):
    detection = NormalizedDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        bounding_box={"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8},
    )
    frame, configuration, batch = sample((detection,))
    with pytest.raises(ValueError, match="class"):
        encode_analysis_frame(preview_item(frame, configuration, batch))


def test_empty_raw_batch_does_not_mean_compliant_or_missing():
    frame, configuration, batch = sample(())
    payload = encode_analysis_frame(preview_item(frame, configuration, batch))
    assert payload["modelObjects"]["detections"] == []
    assert payload["detections"] == []


def test_composed_person_model_preview_preserves_ppe_and_composite_provenance():
    frame, configuration, ppe = sample()
    person = ppe.detections[0].model_copy(update={"class_name": "person"})
    persons = ppe.model_copy(
        update={
            "model_artifact_id": "coco-person",
            "model_sha256": "b" * 64,
            "detections": (person,),
        }
    )

    class Detector:
        def __init__(self, batch):
            self.batch = batch

        async def detect(self, source):
            assert source is frame
            return self.batch

    composite = asyncio.run(
        PersonAnchoredPpeDetector(ppe=Detector(ppe), persons=Detector(persons)).detect(frame)
    )
    payload = encode_analysis_frame(preview_item(frame, configuration, composite))
    objects = payload["modelObjects"]
    assert objects["artifactId"] == "person-anchored-ppe"
    assert objects["artifactSha256"] == composite.model_sha256
    assert composite.model_sha256 not in (ppe.model_sha256, persons.model_sha256)
    assert objects["detections"][0]["className"] == "Person"
    assert (
        objects["detections"][1:]
        == encode_analysis_frame(preview_item(frame, configuration, ppe))["modelObjects"][
            "detections"
        ][1:]
    )
    assert composite.detections[0].class_name == "person"
    assert payload["detections"] == []


@pytest.mark.parametrize(
    "artifact,version,class_id",
    [
        ("synthetic-unified15", "v1", 0),
        ("person-anchored-ppe", "v2", 0),
        ("person-anchored-ppe", "v1", 4),
    ],
)
def test_composed_person_alias_does_not_relax_other_taxonomy(artifact, version, class_id):
    frame, configuration, batch = sample()
    wrong = batch.model_copy(
        update={
            "model_artifact_id": artifact,
            "model_version": version,
            "detections": (
                batch.detections[0].model_copy(
                    update={"class_id": class_id, "class_name": "person"}
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="class"):
        encode_analysis_frame(preview_item(frame, configuration, wrong))


def test_legacy_preview_without_raw_objects_keeps_existing_shape():
    frame, configuration, _ = sample()
    payload = encode_analysis_frame(AnalysisFrame(frame, None, configuration, "ppe", frozenset()))
    assert "modelObjects" not in payload


def test_raw_objects_cannot_override_confirmed_wearer_status_or_create_tracks():
    frame, configuration, batch = sample()
    region_id = configuration.regions[0].region_id
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at_iso,
        frame_dimensions={"width": frame.width, "height": frame.height},
        observations=[
            {
                "type": "PERSON",
                "trackId": 7,
                "boundingBox": {
                    "x1": 0.1,
                    "y1": 0.1,
                    "x2": 0.9,
                    "y2": 0.9,
                    "coordinateSpace": "NORMALIZED_0_1",
                },
            },
            {
                "type": "PPE",
                "trackId": 7,
                "regionId": region_id,
                "geometryVersion": 1,
                "ppeItem": "HARD_HAT",
                "status": "MISSING",
            },
        ],
    )
    common = dict(
        frame=frame,
        observation=event,
        configuration=configuration,
        ppe_region_id=region_id,
        confirmed_ppe_items=frozenset({(7, "HARD_HAT")}),
    )
    baseline = encode_analysis_frame(AnalysisFrame(**common))
    preview = encode_analysis_frame(AnalysisFrame(**common, raw_detections=batch))
    assert len(preview.pop("modelObjects")["detections"]) == 15
    assert preview == baseline
    assert len(preview["detections"]) == 1
    assert preview["detections"][0]["ppeStatus"]["HARD_HAT"] == "MISSING"
    assert preview["detections"][0]["confirmedMissingItems"] == ["HARD_HAT"]


@pytest.mark.parametrize("opted_in", [False, True])
@pytest.mark.anyio
async def test_worker_forwards_exact_raw_detections_only_when_opted_in(
    tmp_path: Path, opted_in: bool
) -> None:
    configuration = _configuration()
    region_id = configuration.regions[0].region_id
    seed = make_test_frame(
        "object-preview", 17, camera_external_id=configuration.camera_external_id
    )
    frames = [
        make_test_frame(
            "object-preview",
            17 + offset,
            session_id=seed.session_id,
            camera_external_id=configuration.camera_external_id,
            captured_at=seed.captured_at + timedelta(seconds=offset),
        )
        for offset in range(3)
    ]
    raw_classes = (
        (0, "Person"),
        (2, "NO-Hardhat"),
        (11, "Mask"),
        (12, "NO-Mask"),
        (13, "fullbody-Harness"),
        (14, "Lanyard-Hook"),
    )
    produced_batches: list[DetectionBatch] = []

    class _Detector:
        async def detect(self, src_frame):
            batch = DetectionBatch.from_frame(
                src_frame,
                model_artifact_id="synthetic-unified15",
                model_version="test-v1",
                model_sha256="a" * 64,
                detections=tuple(
                    NormalizedDetection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=0.9,
                        bounding_box={"x1": 0.35, "y1": 0.1, "x2": 0.65, "y2": 0.5},
                    )
                    for class_id, class_name in raw_classes
                ),
            )
            produced_batches.append(batch)
            return batch

    store = RegionConfigurationStore()
    await store.apply(configuration)
    analyzed: list[AnalysisFrame] = []
    delivered: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        delivered.append(body)
        return httpx.Response(
            202, json={"eventId": body["eventId"], "status": "PROCESSED", "alertIds": []}
        )

    outbox = SqliteEventOutbox((tmp_path / f"worker-{opted_in}.sqlite3").resolve())
    profile = UNIFIED_PPE_15_PROFILE if opted_in else LEGACY_PPE_5_PROFILE
    async with BackendClient(
        "http://backend:3000", "token", transport=httpx.MockTransport(handle)
    ) as client:
        worker = HeadlessCameraProcessingWorker(
            stream=FakeStream(frames),  # type: ignore[arg-type]
            detector=_Detector(),
            pipeline=Mf05Mf06Pipeline(
                tracker=IoUPersonTracker(),
                ppe=PpePipeline.for_model_profile(profile),
                zones=RestrictedZonePipeline(),
                ppe_region_id=region_id,
                schema_version="1.1.0" if opted_in else "1.0.0",
            ),
            ppe_region_id=region_id,
            configurations=store,
            outbox=outbox,
            dispatcher=OutboxDispatcher(outbox, client),
            delivery_interval_seconds=0.01,
            analysis_observer=analyzed.append,
            experimental_object_preview=opted_in,
        )
        result = await worker.run()

    assert result.frames_processed == 3
    assert len(analyzed) == 3
    assert len(delivered) == 1
    ppe_obs = [
        (obs["ppeItem"], obs["status"])
        for obs in delivered[0]["observations"]
        if obs.get("type") == "PPE"
    ]
    assert ppe_obs == [("HARD_HAT", "MISSING")]
    payload = encode_analysis_frame(analyzed[-1])
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["confirmedMissingItems"] == ["HARD_HAT"]
    assert set(payload["detections"][0]["ppeStatus"]).isdisjoint(
        {"MASK", "FULL_BODY_HARNESS", "LANYARD_HOOK"}
    )
    if opted_in:
        assert [item.raw_detections for item in analyzed] == produced_batches
        objects = payload["modelObjects"]
        assert [(d["classId"], d["className"]) for d in objects["detections"]] == list(raw_classes)
    else:
        assert all(item.raw_detections is None for item in analyzed)
        assert "modelObjects" not in payload


@pytest.mark.parametrize(
    "profile,expected",
    [(None, False), ("native-ppe-10", False), ("unified-ppe-15", True)],
)
def test_assemble_sessions_enables_object_preview_only_for_unified_ppe_15(
    tmp_path: Path, profile: str | None, expected: bool
) -> None:
    document = _document(
        tmp_path,
        [
            _entry(
                "gate-01",
                CAMERA_A,
                "CAM-A",
                source={"kind": "path", "path": _video(tmp_path, "cam-a.mp4")},
                outbox=str(tmp_path / "outbox.sqlite3"),
            )
        ],
    )
    bound = bind_manifest(parse_manifest(document), {})
    camera = bound.cameras[0]
    client = SimpleNamespace()
    store = RegionConfigurationStore()
    poller = CameraConfigurationPoller(
        client=client,  # type: ignore[arg-type]
        store=store,
        camera_id=camera.entry.camera_id,
        camera_external_id=camera.camera_external_id,
    )
    previews: dict[str, object] = {}
    sessions = assemble_sessions(
        [
            CameraPlan(
                bound=camera,
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
        experimental_model_profile=profile,  # type: ignore[arg-type]
        observation_schema_version="1.1.0" if profile else "1.0.0",
        preview_observers=previews,  # type: ignore[arg-type]
    )
    assert len(sessions) == 1
    assert sessions[0].worker._experimental_object_preview is expected  # type: ignore[attr-defined]
    assert sessions[0].preview is previews["CAM-A"]


def test_v2_policy_bundle_coexists_with_raw_objects_without_losing_policy_evaluation() -> None:
    bundle = CameraPolicyConfigurationBundle.from_wire_bytes(policy_packet(policy_content()))
    explicit_region_id = bundle.regions[1].region_id
    frame = make_test_frame("v2-policy-preview", 21, camera_external_id=bundle.camera_external_id)
    detections = tuple(
        NormalizedDetection(
            class_id=class_id,
            class_name=class_name,
            confidence=0.9,
            bounding_box={"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8},
        )
        for class_id, class_name in UNIFIED_PPE_15_PROFILE.class_map
    )
    batch = DetectionBatch.from_frame(
        frame,
        model_artifact_id="synthetic-unified15",
        model_version="test-v1",
        model_sha256="a" * 64,
        detections=detections,
    )
    event = TechnicalObservationEvent.create(
        event_id="11111111-1111-4111-8111-111111111111",
        schema_version="1.1.0",
        camera_external_id=frame.camera_external_id,
        stream_session_id=str(frame.session_id),
        captured_at=frame.captured_at_iso,
        frame_dimensions={"width": frame.width, "height": frame.height},
        observations=[
            {
                "type": "PERSON",
                "trackId": 7,
                "boundingBox": {
                    "x1": 0.35,
                    "y1": 0.25,
                    "x2": 0.55,
                    "y2": 0.60,
                    "coordinateSpace": "NORMALIZED_0_1",
                },
            },
            {
                "type": "PPE",
                "trackId": 7,
                "regionId": explicit_region_id,
                "geometryVersion": 1,
                "ppeItem": "GLOVES",
                "status": "MISSING",
            },
        ],
    )
    common = dict(
        frame=frame,
        observation=event,
        configuration=bundle,
        ppe_region_id=explicit_region_id,
        confirmed_ppe_items=frozenset({(7, "GLOVES")}),
    )
    baseline = encode_analysis_frame(AnalysisFrame(**common))
    preview = encode_analysis_frame(AnalysisFrame(**common, raw_detections=batch))
    objects = preview.pop("modelObjects")
    assert objects["scope"] == "OBJECT_DETECTION_ONLY"
    assert len(objects["detections"]) == 15
    assert preview == baseline
    assert preview["detections"][0]["ppeStatus"] == {
        "HARD_HAT": "UNKNOWN",
        "SAFETY_VEST": "UNKNOWN",
        "GLOVES": "MISSING",
    }
    assert preview["detections"][0]["confirmedMissingItems"] == ["GLOVES"]
    assert preview["detections"][0]["alertState"] == "CONFIRMED"
    assert preview["detections"][0]["active"] is True
