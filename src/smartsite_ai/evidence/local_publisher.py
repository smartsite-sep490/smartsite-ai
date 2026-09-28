"""Injectable local filesystem evidence publisher with atomic storage."""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from smartsite_ai.domain.observations import EvidenceItem, TechnicalObservationEvent
from smartsite_ai.evidence.models import (
    EvidenceBindingError,
    EvidenceConflictError,
    EvidenceEncodingError,
    EvidencePathContainmentError,
    EvidenceSizeLimitError,
    build_local_evidence_uri,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.training.dataset_integrity import is_link_like

_SAFE_ID_RE = re.compile(r"^[0-9a-zA-Z_-]+$")


class LocalEvidencePublisher:
    """Publishes exact-frame evidence to provider-neutral immutable local storage.

    Threat Boundary:
        Evidence artifacts are written directly into a dedicated owner-private root directory
        using flat filenames formatted as `{sessionId}_{sequenceNumber}_{eventId}.jpg`.
        This class strictly enforces that the root directory exists, is a regular non-symlink
        directory, and rejects path traversal via identifiers.
        Concurrent OS-level filesystem attacks by local unprivileged actors (e.g. TOCTOU attacks
        swapping root_dir with a junction) are outside this application boundary and must be
        secured by OS-level owner-private permissions (NTFS ACLs).
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        max_jpeg_bytes: int = 1_048_576,
        jpeg_quality: int = 85,
        cv2_module: object | None = None,
    ) -> None:
        if max_jpeg_bytes <= 0:
            raise ValueError("max_jpeg_bytes must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        if not root_dir.exists():
            raise EvidencePathContainmentError(f"evidence root directory must exist: {root_dir}")
        if not root_dir.is_dir():
            raise EvidencePathContainmentError(
                f"evidence root directory must be a directory: {root_dir}"
            )
        if is_link_like(root_dir):
            raise EvidencePathContainmentError(
                f"evidence root directory must not be a symlink: {root_dir}"
            )

        self._root_dir = root_dir.resolve()
        self._max_jpeg_bytes = max_jpeg_bytes
        self._jpeg_quality = jpeg_quality
        self._cv2_module = cv2_module

    def _resolve_evidence_path(
        self,
        *,
        session_id: str,
        sequence_number: int,
        event_id: str,
    ) -> Path:
        """Validate path containment and return target flat image path in root directory."""
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

        filename = f"{session_id}_{sequence_number}_{event_id}.jpg"
        target_image = self._root_dir / filename

        try:
            resolved_image = target_image.resolve()
            if resolved_image.parent != self._root_dir or not resolved_image.is_relative_to(
                self._root_dir
            ):
                raise EvidencePathContainmentError(
                    f"evidence path escapes root boundary {self._root_dir}: {target_image}"
                )
        except (ValueError, RuntimeError) as error:
            raise EvidencePathContainmentError(
                f"evidence path escapes root boundary {self._root_dir}: {target_image}"
            ) from error

        if target_image.exists() and is_link_like(target_image):
            raise EvidencePathContainmentError(
                f"evidence image path must not be a symlink: {target_image}"
            )

        return target_image

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

        # 2. Revalidate root directory before write
        if (
            not self._root_dir.exists()
            or not self._root_dir.is_dir()
            or is_link_like(self._root_dir)
        ):
            raise EvidencePathContainmentError(
                f"evidence root directory is missing or invalid: {self._root_dir}"
            )

        # 3. Path containment (flat filename in root)
        session_id_str = str(frame.session_id)
        image_path = self._resolve_evidence_path(
            session_id=session_id_str,
            sequence_number=frame.sequence_number,
            event_id=event.event_id,
        )

        # 4. Encode JPEG and check size
        jpeg_bytes = self._encode_jpeg(frame)
        if len(jpeg_bytes) > self._max_jpeg_bytes:
            raise EvidenceSizeLimitError(
                f"encoded JPEG size ({len(jpeg_bytes)} bytes) exceeds "
                f"maximum ({self._max_jpeg_bytes} bytes)"
            )

        # 5. Atomic single JPEG write: temp + atomic link/create-if-absent (no-overwrite)
        uri = build_local_evidence_uri(
            session_id=session_id_str,
            sequence_number=frame.sequence_number,
            event_id=event.event_id,
        )

        if image_path.exists():
            existing_bytes = image_path.read_bytes()
            if existing_bytes != jpeg_bytes:
                raise EvidenceConflictError(
                    f"evidence artifact already exists with conflicting content: {image_path}"
                )
        else:
            temp_image: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=self._root_dir, prefix=f".{image_path.name}.", delete=False
                ) as handle:
                    temp_image = Path(handle.name)
                    handle.write(jpeg_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())

                try:
                    os.link(temp_image, image_path)
                    temp_image.unlink(missing_ok=True)
                    temp_image = None
                except FileExistsError as error:
                    temp_image.unlink(missing_ok=True)
                    temp_image = None
                    existing_bytes = image_path.read_bytes()
                    if existing_bytes != jpeg_bytes:
                        raise EvidenceConflictError(
                            "evidence artifact already exists with conflicting content: "
                            f"{image_path}"
                        ) from error
            finally:
                if temp_image is not None:
                    temp_image.unlink(missing_ok=True)

        # 6. Bind reference into event preserving existing items and deduplicating
        evidence_item = EvidenceItem(
            kind="FRAME",
            uri=uri,
        )
        if any(item.uri == uri for item in event.evidence):
            combined_evidence = list(event.evidence)
        else:
            combined_evidence = [*event.evidence, evidence_item]

        return TechnicalObservationEvent.create(
            event_id=event.event_id,
            camera_external_id=event.camera_external_id,
            stream_session_id=event.stream_session_id,
            captured_at=event.captured_at,
            frame_dimensions=event.frame_dimensions,
            observations=event.observations,
            evidence=combined_evidence,
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
