"""Bounded raw-pixel observer of an existing capture owner, independent of YOLO."""

import asyncio
import base64
import copy
import hashlib
import io
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, TypedDict

from smartsite_ai.configuration_image import ConfigurationImagePayload, SourceKind
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.queue import PreviewFrameQueue, QueueClosedError
from smartsite_ai.runtime.supervisor import _finish_cleanup


class CameraMediaFrame(TypedDict):
    type: Literal["media-frame"]
    mediaVersion: Literal[1]
    cameraExternalId: str
    sessionId: str
    sequenceNumber: str
    capturedAt: str
    width: int
    height: int
    imageDataUrl: str


def encode_media_frame(frame: FrameEnvelope) -> CameraMediaFrame:
    """Encode display pixels only; never attach detections from a different frame."""
    if frame.width * frame.height > 16_777_216:
        raise ValueError("Preview input exceeds pixel budget")
    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
    image.thumbnail((640, 640))
    for quality in (65, 45, 25):
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
        data_url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        if len(data_url) <= 180_000:
            return {
                "type": "media-frame",
                "mediaVersion": 1,
                "cameraExternalId": frame.camera_external_id,
                "sessionId": str(frame.session_id),
                "sequenceNumber": str(frame.sequence_number),
                "capturedAt": frame.captured_at_iso,
                "width": image.width,
                "height": image.height,
                "imageDataUrl": data_url,
            }
    raise ValueError("Preview JPEG exceeds byte budget")


class CameraPreviewObserver:
    """One encoder with latest-frame backpressure; HTTP requests never start it."""

    def __init__(
        self,
        camera_external_id: str,
        frames: PreviewFrameQueue,
        *,
        source_kind: SourceKind,
        is_streaming: Callable[[], bool],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        encoder: Callable[[FrameEnvelope], CameraMediaFrame] = encode_media_frame,
        stop_timeout: float = 2.0,
    ) -> None:
        if not camera_external_id or len(camera_external_id) > 128:
            raise ValueError("Preview camera identity is invalid")
        if source_kind not in ("LIVE", "REPLAY"):
            raise ValueError("Preview source kind must be explicit")
        if not math.isfinite(stop_timeout) or not 0 < stop_timeout <= 10:
            raise ValueError("Preview stop budget must be bounded")
        self.camera_external_id = camera_external_id
        self.source_kind = source_kind
        self.state: Literal["IDLE", "RUNNING", "STOPPING", "STOPPED", "FAILED"] = "IDLE"
        self.reason: str | None = None
        self.encodes_completed = 0
        self._frames, self._is_streaming = frames, is_streaming
        self._clock, self._encoder, self._stop_timeout = clock, encoder, stop_timeout
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._latest: CameraMediaFrame | None = None
        self._latest_time: datetime | None = None
        self._revision = -1

    def _fresh(self, captured: datetime) -> bool:
        return -1 <= (self._clock() - captured).total_seconds() < 2

    def latest(self) -> CameraMediaFrame | None:
        if (
            self.state != "RUNNING"
            or self._frames.is_closed
            or not self._is_streaming()
            or self._revision != self._frames.revision
            or self._latest_time is None
            or not self._fresh(self._latest_time)
        ):
            return None
        return copy.deepcopy(self._latest)

    async def start(self) -> None:
        if self.state == "RUNNING":
            return
        if self.state != "IDLE" or self._frames.is_closed:
            raise RuntimeError("Preview observer lifecycle has ended")
        self.state = "RUNNING"
        self._task = asyncio.create_task(self._run(), name="camera-preview-encoder")

    def configuration_image(self) -> ConfigurationImagePayload | None:
        frame = self.latest()
        if frame is None or min(frame["width"], frame["height"]) < 64:
            return None
        jpeg = base64.b64decode(frame["imageDataUrl"].split(",", 1)[1], validate=True)
        return {
            "schemaVersion": 1,
            "purpose": "REGION_CONFIGURATION",
            "sourceKind": self.source_kind,
            "frame": {
                "cameraExternalId": frame["cameraExternalId"],
                "sessionId": frame["sessionId"],
                "sequenceNumber": frame["sequenceNumber"],
                "capturedAt": frame["capturedAt"],
                "width": frame["width"],
                "height": frame["height"],
                "imageDataUrl": frame["imageDataUrl"],
                "imageSha256": hashlib.sha256(jpeg).hexdigest(),
            },
        }

    async def _run(self) -> None:
        try:
            while not self._stopping:
                frame = await self._frames.get()
                revision = self._frames.revision
                if frame.camera_external_id != self.camera_external_id:
                    raise ValueError("Preview camera identity mismatch")
                if not self._fresh(frame.captured_at) or not self._is_streaming():
                    continue
                payload = await _finish_cleanup(asyncio.to_thread(self._encoder, frame))
                self.encodes_completed += 1
                if (
                    not self._stopping
                    and not self._frames.is_closed
                    and self._frames.revision == revision
                    and self._is_streaming()
                    and self._fresh(frame.captured_at)
                ):
                    self._latest = copy.deepcopy(payload)
                    self._latest_time, self._revision = frame.captured_at, revision
                # One encoder and at most ten encodes/second, independent of viewers.
                await asyncio.sleep(0.1)
        except QueueClosedError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            self.state, self.reason = "FAILED", "preview_encoding_failed"
        finally:
            self._latest = self._latest_time = None
            self._frames.close()
            if self.state != "FAILED":
                self.state = "STOPPED"

    async def stop(self) -> bool:
        self._stopping = True
        self._latest = self._latest_time = None
        self._frames.close()
        if self._task is None:
            self.state = "STOPPED"
            return True
        if self.state == "RUNNING":
            self.state = "STOPPING"
        done, _ = await asyncio.wait({self._task}, timeout=self._stop_timeout)
        return bool(done)

    async def wait_for_release(self) -> None:
        if not self._stopping:
            raise RuntimeError("Stop must be requested before draining preview")
        if self._task is not None:
            await _finish_cleanup(self._task)
