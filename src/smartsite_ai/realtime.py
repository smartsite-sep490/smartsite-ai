"""Local realtime stream: verified YOLO detector, MF05/MF06 pipeline, optional backend post."""

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from smartsite_ai.config import Settings
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.artifacts import ModelArtifactSpec
from smartsite_ai.inference.loading import load_detector_artifact
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.inference.yolo import Yolo11Detector
from smartsite_ai.ingestion.config import StreamConfig
from smartsite_ai.ingestion.opencv_source import OpenCvFrameSource
from smartsite_ai.ingestion.source import SourceConnectionError, SourceReadError
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.ppe_temporal import (
    TemporalPpeCandidateGate,
    filter_event_for_delivery,
)
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.runtime_device import validate_runtime_device
from smartsite_ai.tracking.iou_tracker import IoUPersonTracker


def _ui_frame(
    event: Any,
    width: int,
    height: int,
    *,
    confirmed_ppe_items: frozenset[tuple[int, str]] = frozenset(),
    occupied_zone_regions: frozenset[tuple[int, str]] = frozenset(),
) -> dict[str, Any]:
    """Map one technical event to the JSON shape the web realtime screen already reads."""

    if event is None:
        return {
            "type": "frame",
            "width": width,
            "height": height,
            "detections": [],
            "zoneDetections": [],
        }
    payload = event.to_wire_dict()
    people: dict[int, dict[str, Any]] = {}
    ppe_status: dict[int, dict[str, str]] = {}
    entered_zone_regions: set[tuple[int, str]] = set()
    for observation in payload["observations"]:
        track_id = int(observation["trackId"])
        kind = observation["type"]
        if kind == "PERSON":
            people[track_id] = observation
        elif kind == "PPE":
            ppe_status.setdefault(track_id, {})[observation["ppeItem"]] = observation["status"]
        elif kind == "ZONE_ENTRY":
            entered_zone_regions.add((track_id, observation["regionId"]))
    detections = []
    zone_detections = []
    for track_id, person in people.items():
        status = {"HARD_HAT": "UNKNOWN", "SAFETY_VEST": "UNKNOWN"}
        status.update(ppe_status.get(track_id, {}))
        missing = [item for item, value in status.items() if value == "MISSING"]
        confirmed_missing = sorted(
            item for confirmed_track, item in confirmed_ppe_items if confirmed_track == track_id
        )
        if confirmed_missing:
            alert_state = "CONFIRMED"
            label = f"MISSING {confirmed_missing[0].replace('_', ' ')}"
        elif missing:
            alert_state = "PENDING_CONFIRMATION"
            label = "PPE CHECK PENDING"
        elif all(value == "PRESENT" for value in status.values()):
            alert_state = "COMPLIANT"
            label = "PPE COMPLIANT"
        else:
            alert_state = "UNKNOWN"
            label = "PPE UNKNOWN"
        box = person.get("boundingBox")
        detections.append(
            {
                "trackId": track_id,
                "confidence": person.get("confidence") or 0.0,
                "boundingBox": box,
                "ppeStatus": status,
                "alertState": alert_state,
                "confirmedMissingItems": confirmed_missing,
                "active": bool(confirmed_missing),
                "label": label,
            }
        )
        occupied_regions = sorted(
            region_id
            for occupied_track, region_id in occupied_zone_regions
            if occupied_track == track_id
        )
        entered_regions = sorted(
            region_id
            for entered_track, region_id in entered_zone_regions
            if entered_track == track_id
        )
        displayed_regions = sorted(set(occupied_regions) | set(entered_regions))
        if not displayed_regions:
            displayed_regions = [None]
        for region_id in displayed_regions:
            zone_detections.append(
                {
                    "trackId": track_id,
                    "confidence": person.get("confidence") or 0.0,
                    "boundingBox": box,
                    "active": region_id is not None,
                    "label": (
                        "ZONE ENTRY"
                        if region_id in entered_regions
                        else "IN RESTRICTED ZONE"
                        if region_id is not None
                        else "OUTSIDE ZONE"
                    ),
                    "regionId": region_id,
                }
            )
    return {
        "type": "frame",
        "width": width,
        "height": height,
        "cameraExternalId": payload["cameraExternalId"],
        "detections": detections,
        "zoneDetections": zone_detections,
    }


def parse_zone_polygon(raw: str) -> list[tuple[float, float]]:
    """Parse `x,y;x,y` points. Out-of-range values are rejected, not clamped."""

    points: list[tuple[float, float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, y_text = item.split(",", maxsplit=1)
        x = float(x_text)
        y = float(y_text)
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            raise ValueError("zone polygon coordinates must be within 0 and 1")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("zone polygon needs at least 3 points")
    return points


def safe_source_label(source: str) -> str:
    """Hide credentials and local paths from websocket errors."""

    if "://" not in source:
        return "configured source"
    parts = urlsplit(source)
    host = parts.hostname or "source"
    return f"{parts.scheme}://{host}"


def resolve_realtime_device(
    configured: str,
    *,
    cuda_available: bool,
    cuda_device_count: int,
) -> str:
    """Resolve a validated runtime device without silently ignoring explicit CUDA."""

    configured = validate_runtime_device(configured)

    if configured == "cpu":
        return "cpu"
    if configured == "auto":
        return "cuda:0" if cuda_available and cuda_device_count > 0 else "cpu"
    if not cuda_available or cuda_device_count <= 0:
        raise ValueError("CUDA device was requested but CUDA is unavailable")

    device_index = 0 if configured == "cuda" else int(configured.removeprefix("cuda:"))
    if device_index >= cuda_device_count:
        raise ValueError(
            f"CUDA device index {device_index} is unavailable; found {cuda_device_count} device(s)"
        )
    return f"cuda:{device_index}"


def _configured_realtime_device(configured: str) -> str:
    try:
        import torch
    except ImportError:
        return resolve_realtime_device(
            configured,
            cuda_available=False,
            cuda_device_count=0,
        )
    return resolve_realtime_device(
        configured,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
    )


def _load_class_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("class map must be a non-empty object")
    return {int(key): str(value) for key, value in raw.items()}


def _load_realtime_stack(
    settings: Settings,
) -> tuple[Yolo11Detector, UltralyticsYoloRunner, Mf05Mf06Pipeline, CameraRegionConfiguration]:
    if not settings.realtime_model_path or not settings.realtime_class_map_path:
        raise ValueError("Realtime model and class map are not configured")
    if not settings.realtime_region_configuration_path:
        raise ValueError("Realtime region configuration is not configured")
    class_map = _load_class_map(Path(settings.realtime_class_map_path))
    spec = ModelArtifactSpec(
        artifact_id=settings.realtime_model_artifact_id,
        version=settings.realtime_model_version,
        model_family=settings.realtime_model_family,
        artifact_path=Path(settings.realtime_model_path),
        sha256=settings.realtime_model_sha256,
        source_url=settings.realtime_model_source_url,
        license=settings.realtime_model_license,
        class_map=tuple(sorted(class_map.items())),
        confidence_threshold=settings.realtime_confidence,
        iou_threshold=0.45,
        image_size=(640, 640),
        device=_configured_realtime_device(settings.realtime_device),
    )
    detector, loaded_runner, _artifact, _class_map = load_detector_artifact(
        spec,
        runner_factory=UltralyticsYoloRunner,
    )
    if not isinstance(loaded_runner, UltralyticsYoloRunner):
        raise TypeError("realtime detector loader returned an unexpected runner")
    runner = loaded_runner
    configuration = CameraRegionConfiguration.from_wire_bytes(
        Path(settings.realtime_region_configuration_path).read_bytes()
    )
    if configuration.camera_external_id != settings.realtime_camera_external_id:
        raise ValueError("realtime camera id does not match the region configuration")
    pipeline = Mf05Mf06Pipeline(
        tracker=IoUPersonTracker(),
        ppe=PpePipeline(),
        zones=RestrictedZonePipeline(),
        ppe_region_id=settings.realtime_ppe_region_id,
    )
    return detector, runner, pipeline, configuration


class RealtimeStreamGateTracker:
    """Manage TemporalPpeCandidateGate lifecycle across replay stream sessions."""

    def __init__(
        self,
        gate_factory: Callable[[], TemporalPpeCandidateGate] = TemporalPpeCandidateGate,
    ) -> None:
        self._gate_factory = gate_factory
        self._current_session_id: UUID | None = None
        self._gate: TemporalPpeCandidateGate = self._gate_factory()

    @property
    def current_session_id(self) -> UUID | None:
        return self._current_session_id

    @property
    def gate(self) -> TemporalPpeCandidateGate:
        return self._gate

    def get_gate(self, session_id: UUID) -> TemporalPpeCandidateGate:
        if self._current_session_id is not None and self._current_session_id != session_id:
            self._gate = self._gate_factory()
        self._current_session_id = session_id
        return self._gate


async def stream_realtime(websocket: WebSocket, settings: Settings) -> None:
    """Stream one source through the technical pipeline until the client disconnects."""

    await websocket.accept()
    source = settings.realtime_source or ""
    frame_source: OpenCvFrameSource | None = None
    runner: UltralyticsYoloRunner | None = None
    backend: BackendClient | None = None
    try:
        try:
            detector, runner, pipeline, configuration = _load_realtime_stack(settings)
        except Exception:
            await websocket.send_json({"type": "error", "message": "Realtime model is unavailable"})
            return
        if settings.backend_ingestion_url and settings.backend_service_token:
            backend = BackendClient.from_settings(
                settings,
                timeout=httpx.Timeout(2.0),
                max_retries=0,
            )
        replay_file = Path(source).is_file()
        frame_source = OpenCvFrameSource(
            StreamConfig(
                stream_id="realtime",
                camera_external_id=settings.realtime_camera_external_id,
                source_url=source,
                is_live=not replay_file,
                max_consecutive_failures=3,
            )
        )
        try:
            await frame_source.connect()
        except SourceConnectionError:
            await websocket.send_json(
                {"type": "error", "message": f"Cannot open {safe_source_label(source)}"}
            )
            return
        gate_tracker = RealtimeStreamGateTracker()
        failures = 0
        while True:
            try:
                envelope = await frame_source.read_frame()
            except SourceReadError:
                failures += 1
                if failures >= 3:
                    await websocket.send_json(
                        {"type": "error", "message": "Realtime source stopped producing frames"}
                    )
                    return
                await asyncio.sleep(0.2)
                continue
            if envelope is None:
                if replay_file:
                    await frame_source.connect()
                    await asyncio.sleep(0.05)
                    continue
                failures += 1
                if failures >= 3:
                    await websocket.send_json(
                        {"type": "error", "message": "Realtime source stopped producing frames"}
                    )
                    return
                await asyncio.sleep(0.2)
                continue
            failures = 0
            batch = await detector.detect(envelope)
            event = pipeline.process(
                batch,
                region_configuration=configuration,
                event_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"smartsite-realtime:{envelope.session_id}:{envelope.sequence_number}",
                    )
                ),
            )
            active_track_ids: tuple[int, ...] = ()
            ppe_observations: tuple[Any, ...] = ()
            if event is not None:
                active_track_ids = tuple(
                    obs.track_id for obs in event.observations if obs.type == "PERSON"
                )
                ppe_observations = tuple(obs for obs in event.observations if obs.type == "PPE")
            temporal_gate = gate_tracker.get_gate(batch.session_id)
            confirmed_candidates = temporal_gate.update(
                stream_id=batch.stream_id,
                session_id=batch.session_id,
                observed_at=batch.captured_at,
                active_track_ids=active_track_ids,
                observations=ppe_observations,
            )
            confirmed_ppe_items = temporal_gate.confirmed_track_items(
                stream_id=batch.stream_id,
                session_id=batch.session_id,
                active_track_ids=active_track_ids,
            )
            occupied_zone_regions = pipeline.occupied_zone_track_regions(
                stream_id=batch.stream_id,
                session_id=batch.session_id,
                active_track_ids=active_track_ids,
            )
            await websocket.send_json(
                _ui_frame(
                    event,
                    envelope.width,
                    envelope.height,
                    confirmed_ppe_items=confirmed_ppe_items,
                    occupied_zone_regions=occupied_zone_regions,
                )
            )
            if backend is not None and event is not None:
                delivery_event = filter_event_for_delivery(event, confirmed_candidates)
                if delivery_event is not None:
                    with suppress(Exception):
                        await backend.post_event(delivery_event)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        with suppress(Exception):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Realtime inference failed: {type(exc).__name__}",
                }
            )
    finally:
        if runner is not None:
            runner.close()
        if backend is not None:
            with suppress(Exception):
                await backend.aclose()
        if frame_source is not None:
            with suppress(Exception):
                await frame_source.close()
        with suppress(Exception):
            await websocket.close()
