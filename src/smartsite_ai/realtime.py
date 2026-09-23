"""Local realtime stream: verified YOLO detector, MF05/MF06 pipeline, optional backend post."""

import asyncio
import importlib
import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from smartsite_ai.config import Settings
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.artifacts import ModelArtifactSpec, verify_model_artifact
from smartsite_ai.inference.ultralytics_runner import UltralyticsYoloRunner
from smartsite_ai.inference.yolo import Yolo11Detector
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.pipelines.mf05_mf06 import Mf05Mf06Pipeline
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.tracking.iou_tracker import IoUPersonTracker


def _ui_frame(event: Any, width: int, height: int) -> dict[str, Any]:
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
    zone_tracks: set[int] = set()
    for observation in payload["observations"]:
        track_id = int(observation["trackId"])
        kind = observation["type"]
        if kind == "PERSON":
            people[track_id] = observation
        elif kind == "PPE":
            ppe_status.setdefault(track_id, {})[observation["ppeItem"]] = observation["status"]
        elif kind == "ZONE_ENTRY":
            zone_tracks.add(track_id)
    detections = []
    zone_detections = []
    for track_id, person in people.items():
        status = {"HARD_HAT": "UNKNOWN", "SAFETY_VEST": "UNKNOWN"}
        status.update(ppe_status.get(track_id, {}))
        missing = [item for item, value in status.items() if value == "MISSING"]
        box = person.get("boundingBox")
        detections.append(
            {
                "trackId": track_id,
                "confidence": person.get("confidence") or 0.0,
                "boundingBox": box,
                "ppeStatus": status,
                "active": bool(missing),
                "label": f"MISSING {missing[0].replace('_', ' ')}" if missing else "PPE OK",
            }
        )
        zone_detections.append(
            {
                "trackId": track_id,
                "confidence": person.get("confidence") or 0.0,
                "boundingBox": box,
                "active": track_id in zone_tracks,
                "label": "ZONE ENTRY" if track_id in zone_tracks else "OUTSIDE ZONE",
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


def _should_post(event: Any) -> bool:
    payload = event.to_wire_dict()
    return any(
        observation["type"] == "ZONE_ENTRY"
        or (observation["type"] == "PPE" and observation["status"] == "MISSING")
        for observation in payload["observations"]
    )


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
        device="cpu",
    )
    artifact = verify_model_artifact(spec)
    runner = UltralyticsYoloRunner()
    runner.load(artifact)
    detector = Yolo11Detector(artifact, runner)
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


async def stream_realtime(websocket: WebSocket, settings: Settings) -> None:
    """Stream one source through the technical pipeline until the client disconnects."""

    await websocket.accept()
    source = settings.realtime_source or ""
    capture = None
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
        cv2 = importlib.import_module("cv2")
        capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        if not capture.isOpened():
            await websocket.send_json(
                {"type": "error", "message": f"Cannot open {safe_source_label(source)}"}
            )
            return
        replay_file = Path(source).is_file()
        failures = 0
        session_id = uuid4()
        sequence = 0
        while True:
            ok, frame = await asyncio.to_thread(capture.read)
            if not ok or frame is None:
                if replay_file:
                    await asyncio.to_thread(capture.set, cv2.CAP_PROP_POS_FRAMES, 0)
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
            height, width = frame.shape[:2]
            contiguous = frame if frame.flags["C_CONTIGUOUS"] else frame.copy()
            envelope = FrameEnvelope(
                stream_id="realtime",
                session_id=session_id,
                camera_external_id=settings.realtime_camera_external_id,
                captured_at=datetime.now(UTC),
                width=width,
                height=height,
                sequence_number=sequence,
                payload=contiguous.tobytes(),
            )
            batch = await detector.detect(envelope)
            event = pipeline.process(
                batch,
                region_configuration=configuration,
                event_id=str(uuid5(NAMESPACE_URL, f"smartsite-realtime:{session_id}:{sequence}")),
            )
            await websocket.send_json(_ui_frame(event, width, height))
            if backend is not None and event is not None and _should_post(event):
                with suppress(Exception):
                    await backend.post_event(event)
            sequence += 1
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
        if capture is not None:
            with suppress(Exception):
                capture.release()
        with suppress(Exception):
            await websocket.close()
