"""Latest exact analyzed-frame preview, separate from raw camera media."""

import asyncio
import copy
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle
from smartsite_ai.inference.ppe_profiles import UNIFIED_PPE_15_PROFILE
from smartsite_ai.pipelines._geometry import point_in_polygon, to_observation_bounding_box
from smartsite_ai.pipelines.analysis_frame import AnalysisFrame
from smartsite_ai.processing_worker import validate_frame_batch_binding
from smartsite_ai.runtime.supervisor import _finish_cleanup


def encode_analysis_frame(item: AnalysisFrame) -> dict[str, Any]:
    if item.zone_anchor not in ("BOTTOM_CENTER", "BOX_CENTER"):
        raise ValueError("Invalid Zone preview anchor")
    raw = item.raw_detections
    if raw is not None:
        validate_frame_batch_binding(item.frame, raw)
        class_map = dict(UNIFIED_PPE_15_PROFILE.class_map)
        # The explicit two-artifact adapter retains COCO's lowercase Person label.
        object_names = [
            "Person"
            if (raw.model_artifact_id, raw.model_version, d.class_id, d.class_name)
            == ("person-anchored-ppe", "v1", 0, "person")
            else d.class_name
            for d in raw.detections
        ]
        if any(
            class_map.get(d.class_id) != name
            for d, name in zip(raw.detections, object_names, strict=True)
        ):
            raise ValueError("Raw preview class mapping does not match unified-ppe-15")
    # Keep the existing technical preview mapping, without starting its socket
    # runtime, model, camera, or Backend delivery loop.
    from smartsite_ai.realtime import build_realtime_preview

    payload = build_realtime_preview(
        item.observation,
        item.frame,
        configuration=item.configuration,
        ppe_region_id=item.ppe_region_id,
        confirmed_ppe_items=item.confirmed_ppe_items,
        occupied_zone_regions=item.occupied_zone_regions,
    )
    if raw is not None:
        payload["modelObjects"] = {
            "modelFamily": "yolo11s",
            "experimental": True,
            "scope": "OBJECT_DETECTION_ONLY",
            "artifactId": raw.model_artifact_id,
            "artifactSha256": raw.model_sha256,
            "detections": [
                {
                    "classId": d.class_id,
                    "className": name,
                    "confidence": d.confidence,
                    "boundingBox": to_observation_bounding_box(d.bounding_box).model_dump(
                        by_alias=True
                    ),
                }
                for d, name in zip(raw.detections, object_names, strict=True)
            ],
        }
    if item.zone_anchor == "BOX_CENTER":
        payload["zoneAnchor"] = item.zone_anchor
    if isinstance(item.configuration, CameraPolicyConfigurationBundle):
        for person in payload["detections"]:
            box = person.get("boundingBox")
            if not box:
                person.update(alertState="UNKNOWN", label="PPE UNKNOWN", active=False)
                continue
            point = ((box["x1"] + box["x2"]) / 2, box["y2"])
            regions = tuple(
                region.region_id
                for region in item.configuration.regions
                if point_in_polygon(point, region.polygon.coordinates)
            )
            required = item.configuration.resolve_ppe_policy(regions).required_ppe
            observed = person["ppeStatus"]
            status = {ppe: observed.get(ppe, "UNKNOWN") for ppe in required}
            missing = sorted(ppe for ppe, state in status.items() if state == "MISSING")
            confirmed = sorted(ppe for ppe in person["confirmedMissingItems"] if ppe in required)
            person["ppeStatus"] = status
            person["confirmedMissingItems"] = confirmed
            person["active"] = bool(confirmed)
            if confirmed:
                person.update(
                    alertState="CONFIRMED", label="MISSING " + confirmed[0].replace("_", " ")
                )
            elif missing:
                person.update(alertState="PENDING_CONFIRMATION", label="PPE CHECK PENDING")
            elif required and all(state == "PRESENT" for state in status.values()):
                person.update(alertState="COMPLIANT", label="PPE COMPLIANT")
            else:
                person.update(
                    alertState="UNKNOWN", label="PPE UNKNOWN" if required else "PPE CHECK DISABLED"
                )
    return payload


class AnalysisPreviewObserver:
    """One pending analysis packet and one drained encoder, never an unbounded backlog."""

    def __init__(
        self,
        camera_external_id: str,
        *,
        current: Callable[[AnalysisFrame], Awaitable[bool]],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        encoder: Callable[[AnalysisFrame], dict[str, Any]] = encode_analysis_frame,
    ) -> None:
        self.camera_external_id = camera_external_id
        self._current, self._clock, self._encoder = current, clock, encoder
        self._pending: AnalysisFrame | None = None
        self._latest: tuple[AnalysisFrame, dict[str, Any]] | None = None
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.state = "IDLE"
        self.reason: str | None = None

    def _fresh(self, item: AnalysisFrame) -> bool:
        age = (self._clock() - item.frame.captured_at).total_seconds()
        return item.frame.camera_external_id == self.camera_external_id and -1 <= age < 2

    def offer(self, item: AnalysisFrame) -> None:
        if self.state != "RUNNING" or not self._fresh(item):
            return
        event = item.observation
        if event is not None and (
            event.camera_external_id != item.frame.camera_external_id
            or event.stream_session_id != str(item.frame.session_id)
            or datetime.fromisoformat(event.captured_at) != item.frame.captured_at
            or event.frame_dimensions.width != item.frame.width
            or event.frame_dimensions.height != item.frame.height
        ):
            self._latest = self._pending = None
            self.reason = "analysis_frame_binding_invalid"
            return
        self._pending = item
        self._ready.set()

    async def latest(self) -> dict[str, Any] | None:
        if self.state != "RUNNING" or self._latest is None:
            return None
        item, payload = self._latest
        valid = await self._current(item)
        return (
            copy.deepcopy(payload)
            if valid and self._fresh(item) and self.state == "RUNNING"
            else None
        )

    async def start(self) -> None:
        if self.state == "RUNNING":
            return
        if self.state != "IDLE":
            raise RuntimeError("Analysis observer lifecycle has ended")
        self.state = "RUNNING"
        self._task = asyncio.create_task(self._run(), name="camera-analysis-preview")

    async def _run(self) -> None:
        try:
            while not self._stopping:
                await self._ready.wait()
                self._ready.clear()
                item, self._pending = self._pending, None
                if item is None or not self._fresh(item) or not await self._current(item):
                    continue
                payload = await _finish_cleanup(asyncio.to_thread(self._encoder, item))
                if not self._stopping and await self._current(item) and self._fresh(item):
                    self._latest = item, copy.deepcopy(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.state, self.reason = "FAILED", "analysis_preview_encoding_failed"
        finally:
            self._latest = self._pending = None
            if self.state != "FAILED":
                self.state = "STOPPED"

    async def stop(self) -> bool:
        self._stopping = True
        self._latest = self._pending = None
        self._ready.set()
        if self._task is None:
            self.state = "STOPPED"
            return True
        done, _ = await asyncio.wait({self._task}, timeout=2)
        return bool(done)

    async def wait_for_release(self) -> None:
        if not self._stopping:
            raise RuntimeError("Stop must be requested before draining analysis preview")
        if self._task is not None:
            await _finish_cleanup(self._task)
