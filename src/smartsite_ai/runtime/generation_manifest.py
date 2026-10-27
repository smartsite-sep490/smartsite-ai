"""One-camera supervisor binding without rewriting a reviewed runtime manifest."""

import os
from pathlib import Path
from uuid import UUID

from smartsite_ai.runtime.manifest import CameraRuntimeError, CameraRuntimeManifest, parse_manifest


def bind_generation_manifest(
    manifest: CameraRuntimeManifest,
    credential: Path | None,
    camera_id: UUID | None,
    outbox: Path | None,
) -> CameraRuntimeManifest:
    if credential is None and camera_id is None and outbox is None:
        return manifest
    if (
        credential is None
        or camera_id is None
        or outbox is None
        or not credential.is_absolute()
        or not outbox.is_absolute()
        or len(manifest.cameras) != 1
    ):
        raise CameraRuntimeError("Supervisor binding must identify one camera and absolute paths")
    entry = manifest.cameras[0]
    if entry.camera_id != camera_id or os.path.normcase(
        os.path.normpath(entry.outbox)
    ) != os.path.normcase(os.path.normpath(str(outbox))):
        raise CameraRuntimeError("Supervisor camera or outbox binding mismatch")
    document = manifest.model_dump(by_alias=True, mode="json")
    document["cameras"][0]["generationCredentialFile"] = str(credential)
    return parse_manifest(document)
