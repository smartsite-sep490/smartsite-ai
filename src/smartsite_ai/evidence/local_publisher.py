"""Injectable local filesystem evidence publisher with atomic storage and bounded retention."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from smartsite_ai.domain.observations import EvidenceItem, TechnicalObservationEvent
from smartsite_ai.evidence.models import (
    EvidenceBindingError,
    EvidenceEncodingError,
    EvidenceManifest,
    EvidencePathContainmentError,
    EvidenceSizeLimitError,
    build_local_evidence_uri,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.training.dataset_integrity import is_link_like

_SAFE_ID_RE = re.compile(r"^[0-9a-zA-Z_-]+$")


class LocalEvidencePublisher:
    """Publishes exact-frame evidence to provider-neutral local storage with atomic replace."""

    def __init__(
        self,
        *,
        root_dir: Path,
        max_jpeg_bytes: int = 1_048_576,
        jpeg_quality: int = 85,
        retention_limit: int | None = None,
        cv2_module: object | None = None,
    ) -> None:
        if max_jpeg_bytes <= 0:
            raise ValueError("max_jpeg_bytes must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if retention_limit is not None and retention_limit <= 0:
            raise ValueError("retention_limit must be positive if provided")

        self._root_dir = root_dir.resolve()
        self._max_jpeg_bytes = max_jpeg_bytes
        self._jpeg_quality = jpeg_quality
        self._retention_limit = retention_limit
        self._cv2_module = cv2_module

    def _resolve_evidence_paths(
        self,
        *,
        session_id: str,
        sequence_number: int,
        event_id: str,
    ) -> tuple[Path, Path]:
        """Validate path containment and return target image and manifest paths."""
        if not _SAFE_ID_RE.fullmatch(session_id):
            raise EvidencePathContainmentError(
                f"session_id contains unsafe path characters: {session_id}"
            )
        if not _SAFE_ID_RE.fullmatch(event_id):
            raise EvidencePathContainmentError(
                f"event_id contains unsafe path characters: {event_id}"
            )
        if sequence_number < 0:
            raise EvidencePathContainmentError(
                f"sequence_number must be non-negative: {sequence_number}"
            )

        target_dir = self._root_dir / session_id
        target_image = target_dir / f"{sequence_number}_{event_id}.jpg"
        target_manifest = target_dir / f"{sequence_number}_{event_id}.manifest.json"

        try:
            resolved_dir = target_dir.resolve()
            resolved_image = target_image.resolve()
            resolved_manifest = target_manifest.resolve()
            resolved_dir.relative_to(self._root_dir)
            resolved_image.relative_to(self._root_dir)
            resolved_manifest.relative_to(self._root_dir)
        except (ValueError, RuntimeError) as error:
            raise EvidencePathContainmentError(
                f"evidence path escapes root boundary {self._root_dir}: {target_image}"
            ) from error

        if target_dir.exists() and is_link_like(target_dir):
            raise EvidencePathContainmentError(
                f"evidence directory must not be a symlink: {target_dir}"
            )

        return target_image, target_manifest

    def _encode_jpeg(self, frame: FrameEnvelope) -> bytes:
        """Encode BGR24 frame envelope bytes to JPEG."""
        if self._cv2_module is not None:
            cv2: Any = self._cv2_module
            import numpy as np

            image = np.frombuffer(frame.payload, dtype=np.uint8).reshape(
                (frame.height, frame.width, 3)
            )
            success, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            )
            if not success:
                raise EvidenceEncodingError("OpenCV failed to encode JPEG")
            return bytes(encoded)

        # Standard pillow encoding from packed BGR24
        try:
            image = Image.frombytes("RGB", (frame.width, frame.height), frame.payload, "raw", "BGR")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._jpeg_quality)
            return buffer.getvalue()
        except Exception as error:
            raise EvidenceEncodingError(f"Pillow failed to encode JPEG: {error}") from error

    def _prune_retention(self) -> None:
        """Prune oldest evidence items if retention limit is exceeded."""
        if self._retention_limit is None or not self._root_dir.is_dir():
            return

        all_images = sorted(
            self._root_dir.glob("*/*.jpg"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        if len(all_images) <= self._retention_limit:
            return

        excess_count = len(all_images) - self._retention_limit
        for image_path in all_images[:excess_count]:
            image_path.unlink(missing_ok=True)
            manifest_path = image_path.with_suffix(".manifest.json")
            manifest_path.unlink(missing_ok=True)
            parent_dir = image_path.parent
            if parent_dir.is_dir() and not any(parent_dir.iterdir()):
                with contextlib.suppress(OSError):
                    parent_dir.rmdir()

    def _publish_sync(
        self,
        frame: FrameEnvelope,
        event: TechnicalObservationEvent,
    ) -> TechnicalObservationEvent:
        # 1. Exact-frame binding checks
        if str(frame.session_id) != event.stream_session_id:
            raise EvidenceBindingError(
                f"frame session_id {frame.session_id} does not match "
                f"event streamSessionId {event.stream_session_id}"
            )
        if frame.camera_external_id != event.camera_external_id:
            raise EvidenceBindingError(
                f"frame camera_external_id {frame.camera_external_id} does not match "
                f"event cameraExternalId {event.camera_external_id}"
            )
        if (
            frame.width != event.frame_dimensions.width
            or frame.height != event.frame_dimensions.height
        ):
            raise EvidenceBindingError(
                f"frame dimensions ({frame.width}x{frame.height}) do not match "
                f"event ({event.frame_dimensions.width}x{event.frame_dimensions.height})"
            )
        if frame.captured_at.isoformat() != event.captured_at:
            raise EvidenceBindingError(
                f"frame captured_at {frame.captured_at.isoformat()} does not match "
                f"event capturedAt {event.captured_at}"
            )

        # 2. Path containment
        session_id_str = str(frame.session_id)
        image_path, manifest_path = self._resolve_evidence_paths(
            session_id=session_id_str,
            sequence_number=frame.sequence_number,
            event_id=event.event_id,
        )

        # 3. Encode JPEG and check size
        jpeg_bytes = self._encode_jpeg(frame)
        if len(jpeg_bytes) > self._max_jpeg_bytes:
            raise EvidenceSizeLimitError(
                f"encoded JPEG size ({len(jpeg_bytes)} bytes) exceeds "
                f"maximum ({self._max_jpeg_bytes} bytes)"
            )

        # 4. Atomic write temp + replace
        target_dir = image_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        uri = build_local_evidence_uri(
            session_id=session_id_str,
            sequence_number=frame.sequence_number,
            event_id=event.event_id,
        )
        sha256_hash = hashlib.sha256(jpeg_bytes).hexdigest()

        manifest = EvidenceManifest.create(
            event_id=event.event_id,
            stream_session_id=session_id_str,
            sequence_number=frame.sequence_number,
            camera_external_id=frame.camera_external_id,
            captured_at=frame.captured_at.isoformat(),
            uri=uri,
            sha256=sha256_hash,
            size_bytes=len(jpeg_bytes),
            width=frame.width,
            height=frame.height,
        )

        temp_image: Path | None = None
        temp_manifest: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=target_dir, prefix=f".{image_path.name}.", delete=False
            ) as handle:
                temp_image = Path(handle.name)
                handle.write(jpeg_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target_dir,
                prefix=f".{manifest_path.name}.",
                delete=False,
            ) as handle:
                temp_manifest = Path(handle.name)
                json.dump(
                    manifest.to_wire_dict(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            # Atomic replace
            temp_image.replace(image_path)
            temp_image = None
            temp_manifest.replace(manifest_path)
            temp_manifest = None
        finally:
            if temp_image is not None:
                temp_image.unlink(missing_ok=True)
            if temp_manifest is not None:
                temp_manifest.unlink(missing_ok=True)

        # 5. Bound retention
        self._prune_retention()

        # 6. Bind reference into event
        evidence_item = EvidenceItem(
            kind="FRAME",
            uri=uri,
        )
        return TechnicalObservationEvent.create(
            event_id=event.event_id,
            camera_external_id=event.camera_external_id,
            stream_session_id=event.stream_session_id,
            captured_at=event.captured_at,
            frame_dimensions=event.frame_dimensions,
            observations=event.observations,
            evidence=[evidence_item],
            schema_version=event.schema_version,
        )

    async def publish(
        self,
        *,
        frame: FrameEnvelope,
        event: TechnicalObservationEvent,
    ) -> TechnicalObservationEvent:
        """Asynchronously publish frame evidence using a worker thread for disk operations."""
        return await asyncio.to_thread(self._publish_sync, frame, event)


__all__ = ["LocalEvidencePublisher"]
