"""Explicit resume-only binding for the reviewed single-camera runtime command."""

import json
import os
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import SecretStr

from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential
from smartsite_ai.runtime.manifest import CameraRuntimeError, CameraRuntimeManifest, parse_manifest
from smartsite_ai.source_control_models import SourceDraft, SourcePolicy, SourceRejected
from smartsite_ai.source_resume_models import ResumeManifest
from smartsite_ai.training.dataset_integrity import is_link_like

SOURCE_ENV = "SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"
GENERATION_ENV = "SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"


def _require_live_source(source: str, policy: SourcePolicy) -> None:
    for key in policy.webcams:
        if policy.resolve(SourceDraft(kind="WEBCAM", deviceId=key)) == source:
            return
    url = urlsplit(source)
    host = url.hostname or ""
    if ":" in host:
        host = "[" + host + "]"
    authority = host + (":" + str(url.port) if url.port is not None else "")
    endpoint = urlunsplit((url.scheme, authority, url.path, url.query, url.fragment))
    draft = SourceDraft(
        kind="RTSP",
        endpoint=SecretStr(endpoint),
        username=SecretStr(unquote(url.username)) if url.username is not None else None,
        password=SecretStr(unquote(url.password)) if url.password is not None else None,
    )
    # Do not pass absent optional credentials: SourcePolicy validates provided fields too.
    draft = SourceDraft.model_validate(draft.model_dump(exclude_none=True))
    if policy.resolve(draft) != source:
        raise SourceRejected()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = dict(pairs)
    if len(document) != len(pairs):
        raise ValueError("Duplicate runtime credential field")
    return document


def _persist_credential(path: Path, credential: WorkerGenerationCredential) -> None:
    if os.name == "nt":
        raise ValueError("Private credential publication on Windows is not supported")
    # Only replace an existing, private, reviewed regular file; never select a path from env.
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(is_link_like(part) for part in (path, *path.parents))
    ):
        raise ValueError("Unsafe runtime credential path")
    original = path.lstat()
    if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
        raise ValueError("Unsafe runtime credential file")
    if original.st_mode & 0o077 or original.st_uid != os.getuid():
        raise ValueError("Runtime credential file must remain private")
    temporary = path.with_name(path.name + "." + str(uuid4()) + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            document = credential.model_dump(mode="json", by_alias=True)
            document["secret"] = credential.secret.get_secret_value()
            json.dump(document, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        current = path.lstat()
        if (current.st_dev, current.st_ino, current.st_nlink) != (
            original.st_dev,
            original.st_ino,
            1,
        ):
            raise ValueError("Runtime credential owner changed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bind_source_control_child(
    manifest: CameraRuntimeManifest,
    *,
    policy: SourcePolicy,
    scopes: set[tuple[str, str]],
    camera_id: UUID,
    environ: Mapping[str, str],
    expected: ResumeManifest,
) -> CameraRuntimeManifest:
    """Consume trusted driver env; no capture/model/Backend/DB calls and no bootstrap adoption.

    Source replacement and fresh-reference authoring are separate. This profile requires an
    accepted bundle, one tracked LIVE camera, and its already-reviewed credential destination.
    A failed later startup leaves the issued credential in place for explicit reconciliation.
    """
    try:
        if len(manifest.cameras) != 1:
            raise ValueError()
        entry = manifest.cameras[0]
        if (
            entry.camera_id != camera_id
            or not entry.live
            or entry.tracked_v17 is None
            or entry.generation_credential_file is None
            or (str(entry.tracked_v17.site_id), entry.camera_external_id) not in scopes
            and (str(entry.tracked_v17.site_id), "*") not in scopes
        ):
            raise ValueError()
        expected = ResumeManifest.model_validate(expected.model_dump())
        source, raw = environ[SOURCE_ENV], environ[GENERATION_ENV]
        if not source or len(source) > 4096 or len(raw.encode("utf-8")) > 4096:
            raise ValueError()
        _require_live_source(source, policy)
        generation = json.loads(raw, object_pairs_hook=_unique_fields)
        if (
            not isinstance(generation, dict)
            or set(generation) != {"workerGenerationId", "streamSessionId", "secret", "expiresAt"}
            or any(not isinstance(value, str) for value in generation.values())
        ):
            raise ValueError()
        credential = WorkerGenerationCredential.model_validate(
            {
                "schemaVersion": "1.0.0",
                "cameraId": str(camera_id),
                **generation,
            }
        )
        credential.ensure_current(datetime.now(UTC))
        binding = expected.generationBinding
        if (
            binding.cameraId != credential.camera_id
            or binding.workerGenerationId != credential.worker_generation_id
            or binding.streamSessionId != credential.stream_session_id
        ):
            raise ValueError()
        document = manifest.model_dump(mode="json", by_alias=True)
        document["cameras"][0]["source"] = {"kind": "env", "name": SOURCE_ENV}
        bound = parse_manifest(document)
        _persist_credential(Path(entry.generation_credential_file), credential)
        return bound
    except (ValueError, RuntimeError, OSError, KeyError, TypeError):
        raise CameraRuntimeError("Source-control child binding is unavailable or invalid") from None
