"""Small local YOLO tracking stream used by the web realtime test screen."""

import asyncio
import importlib
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect

from smartsite_ai.inference.ultralytics_runner import _default_model_factory


def _box(values: Any, width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = (float(value) for value in values)
    return {
        "x1": max(0.0, min(1.0, x1 / width)),
        "y1": max(0.0, min(1.0, y1 / height)),
        "x2": max(0.0, min(1.0, x2 / width)),
        "y2": max(0.0, min(1.0, y2 / height)),
    }


def _center(box: dict[str, float]) -> tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def _inside(point: tuple[float, float], person: dict[str, float]) -> bool:
    return person["x1"] <= point[0] <= person["x2"] and person["y1"] <= point[1] <= person["y2"]


def _inside_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < crossing:
                inside = not inside
        previous = current
    return inside


def _read_frame(
    model: Any,
    capture: Any,
    confidence: float,
    zone_polygon: list[tuple[float, float]],
) -> dict[str, Any] | None:
    ok, frame = capture.read()
    if not ok or frame is None:
        return None

    height, width = frame.shape[:2]
    result = model.track(frame, persist=True, conf=confidence, verbose=False)[0]
    names = result.names
    boxes = result.boxes
    raw = []
    if boxes is None:
        return {
            "type": "frame",
            "width": width,
            "height": height,
            "detections": [],
            "zoneDetections": [],
        }
    for index in range(len(boxes)):
        xyxy = boxes.xyxy[index].tolist()
        class_id = int(boxes.cls[index].item())
        class_name = str(names[class_id])
        track_id = int(boxes.id[index].item()) if boxes.id is not None else index + 1
        normalized = _box(xyxy, width, height)
        raw.append(
            {
                "trackId": track_id,
                "className": class_name,
                "confidence": float(boxes.conf[index].item()),
                "boundingBox": normalized,
            }
        )

    persons = [item for item in raw if item["className"].casefold() in {"person", "worker"}]
    equipment = [item for item in raw if item not in persons]
    detections = []
    zone_detections = []
    for person in persons:
        status = {"HARD_HAT": "UNKNOWN", "SAFETY_VEST": "UNKNOWN"}
        for item in equipment:
            if not _inside(_center(item["boundingBox"]), person["boundingBox"]):
                continue
            name = item["className"].casefold().replace("_", "-")
            if "hardhat" in name or "hard-hat" in name or "helmet" in name:
                status["HARD_HAT"] = "MISSING" if name.startswith(("no-", "no ")) else "PRESENT"
            if "safety vest" in name or "safety-vest" in name or name in {"vest", "no-vest"}:
                status["SAFETY_VEST"] = "MISSING" if name.startswith(("no-", "no ")) else "PRESENT"
        missing = [item for item, value in status.items() if value == "MISSING"]
        detections.append(
            {
                "trackId": person["trackId"],
                "confidence": person["confidence"],
                "boundingBox": person["boundingBox"],
                "ppeStatus": status,
                "active": bool(missing),
                "label": f"MISSING {missing[0].replace('_', ' ')}" if missing else "PPE OK",
            }
        )
        feet = (
            (person["boundingBox"]["x1"] + person["boundingBox"]["x2"]) / 2,
            person["boundingBox"]["y2"],
        )
        zone_active = _inside_polygon(feet, zone_polygon)
        zone_detections.append(
            {
                "trackId": person["trackId"],
                "confidence": person["confidence"],
                "boundingBox": person["boundingBox"],
                "active": zone_active,
                "label": "ZONE ENTRY" if zone_active else "OUTSIDE ZONE",
            }
        )

    return {
        "type": "frame",
        "width": width,
        "height": height,
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


async def stream_realtime(
    websocket: WebSocket,
    *,
    model_path: str,
    source: str,
    confidence: float,
    zone_polygon: list[tuple[float, float]],
) -> None:
    """Stream one local MP4/RTSP source until the client disconnects."""
    await websocket.accept()
    capture = None
    try:
        model_file = Path(model_path)
        try:
            model = _default_model_factory(model_file)
        except Exception:
            await websocket.send_json({"type": "error", "message": "Realtime model is unavailable"})
            return
        cv2 = importlib.import_module("cv2")
        capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        if not capture.isOpened():
            await websocket.send_json(
                {"type": "error", "message": f"Cannot open {safe_source_label(source)}"}
            )
            return
        failures = 0
        while True:
            payload = await asyncio.to_thread(_read_frame, model, capture, confidence, zone_polygon)
            if payload is None:
                failures += 1
                if failures >= 3:
                    await websocket.send_json(
                        {"type": "error", "message": "Realtime source stopped producing frames"}
                    )
                    return
                await asyncio.sleep(0.2)
                continue
            failures = 0
            await websocket.send_json(payload)
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
        if capture is not None:
            with suppress(Exception):
                capture.release()
        with suppress(Exception):
            await websocket.close()
