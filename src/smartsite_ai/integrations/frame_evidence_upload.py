"""Read only the original event-bound JPEG from an operator-private evidence root."""

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

from smartsite_ai.domain.configuration_bundle import _reject_duplicate_json_keys
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.tracked_observations import TrackedObservationEvent
from smartsite_ai.integrations.backend_client import BackendClient
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.training.dataset_integrity import is_link_like


class TrackedFrameEvidenceUploader:
    def __init__(self, backend: BackendClient, root: Path) -> None:
        if not root.is_absolute() or is_link_like(root) or not root.is_dir():
            raise ValueError("Private evidence root is unavailable")
        self._root = root.resolve(strict=True)
        self._backend = backend

    async def __call__(self, credential: WorkerGenerationCredential, wire: bytes) -> None:
        if not isinstance(wire, bytes) or len(wire) > 1_048_576:
            raise ValueError("Tracked evidence envelope is invalid")
        try:
            envelope = json.loads(wire, object_pairs_hook=_reject_duplicate_json_keys)
            if set(envelope) != {"schemaVersion", "event", "context"} or envelope["schemaVersion"] != "1.0.0":
                raise ValueError()
            event = envelope["event"]
            TrackedObservationEvent.model_validate(event)
            context = EventDeliveryContext.model_validate(envelope["context"])
            if context.source_kind != "LIVE":
                raise ValueError()
            frames = [item for item in event["evidence"] if item["kind"] == "FRAME"]
            session = event["streamSessionId"].lower()
            event_id = event["eventId"].lower()
            sequence = event["framePartition"]["sequenceNumber"]
            if session != context.stream_session_id or session != credential.stream_session_id:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ValueError("Tracked evidence envelope is invalid") from None
        for frame in frames:
            expected_uri = f"local://evidence/{session}/{sequence}/{event_id}.jpg"
            if frame["uri"] != expected_uri:
                raise ValueError("Tracked evidence reference is invalid")
            name = f"{session}_{sequence}_{event_id}.jpg"
            image = await asyncio.to_thread(self._read, name, frame["sha256"])
            await self._backend.upload_tracked_frame(credential, wire, image)

    def _read(self, name: str, digest: str) -> bytes:
        try:
            if is_link_like(self._root) or not self._root.is_dir():
                raise ValueError()
            target = self._root / name
            if is_link_like(target) or target.resolve(strict=True).parent != self._root:
                raise ValueError()
            before = target.stat()
            descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or not 4 <= opened.st_size <= 1_048_576:
                    raise ValueError()
                if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ValueError()
                image = handle.read(1_048_577)
                after = os.fstat(handle.fileno())
            if (
                len(image) != opened.st_size
                or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                or hashlib.sha256(image).hexdigest() != digest
            ):
                raise ValueError()
            return image
        except (OSError, ValueError):
            raise ValueError("Original tracked evidence is unavailable or inconsistent") from None
