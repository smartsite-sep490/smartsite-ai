"""One explicitly configured camera process, with no adoption of unknown owners."""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

from smartsite_ai.ingestion.configuration_process_lease import ConfigurationProcessLease
from smartsite_ai.source_activation import Scope, SourceActivationError
from smartsite_ai.source_resume_models import ResumeBinding, ResumeConfiguration, ResumeManifest
from smartsite_ai.training.dataset_integrity import DatasetIntegrityError, is_link_like

_MAX_GENERATION_FILE_BYTES = 16_384


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


class SourceProcessDriver:
    def __init__(
        self,
        *,
        scope: Scope,
        command: tuple[str, ...],
        stop_file: Path,
        generation_file: Path,
        environment: Mapping[str, str],
        ready: Callable[[str], Awaitable[bool]],
        allow_unowned_start: bool = False,
        resume_command: tuple[str, ...] | None = None,
        resume_camera_id: str | None = None,
        resume_manifest_supported: bool = False,
        resume_configuration_loader: Callable[
            [ResumeManifest, Mapping[str, str]], Awaitable[ResumeConfiguration]
        ]
        | None = None,
        capture_lease_directory: Path | None = None,
        capture_lease_resource_id: UUID | None = None,
    ) -> None:
        inference_command = command if resume_command is None else resume_command
        if (
            not command
            or not Path(command[0]).is_absolute()
            or not inference_command
            or not Path(inference_command[0]).is_absolute()
            or not stop_file.is_absolute()
            or not generation_file.is_absolute()
        ):
            raise ValueError("Reviewed absolute runtime paths required")
        if (capture_lease_directory is None) != (capture_lease_resource_id is None):
            raise ValueError(
                "Stopped-owner recovery requires both capture lease directory and resource UUID"
            )
        if capture_lease_directory is not None and capture_lease_resource_id is not None:
            ConfigurationProcessLease(capture_lease_directory, capture_lease_resource_id)
        forbidden = {"DATABASE_URL", "TEST_DATABASE_URL", "PGPASSWORD", "PGUSER"}
        if forbidden.intersection(k.upper() for k in environment):
            raise ValueError("Worker must not receive database credentials")
        self.scope = scope
        self.command = command
        self.resume_command = inference_command
        self.stop_file = stop_file
        self.generation_file = generation_file
        self.environment = dict(environment)
        self.ready = ready
        self.allow_unowned_start = allow_unowned_start is True
        if (resume_camera_id is None) != (resume_configuration_loader is None):
            raise ValueError("Resume requires a Camera binding and guarded configuration loader")
        self.resume_camera_id = str(UUID(resume_camera_id)) if resume_camera_id else None
        self.resume_manifest_supported = resume_manifest_supported
        # Opt-in asserts that the reviewed command also consumes the fixed manifest env below
        # and verifies it before model/capture startup; an arbitrary legacy command is insufficient.
        self.resume_configuration_loader = resume_configuration_loader
        self.capture_lease_directory = capture_lease_directory
        self.capture_lease_resource_id = capture_lease_resource_id
        self.process: asyncio.subprocess.Process | None = None
        self._current_source: str | None = None
        self._owned_generation: dict[str, str] | None = None
        self._reconciliation_required = False

    @property
    def current_source(self) -> str | None:
        if self.process is None or self.process.returncode is not None:
            return None
        return self._current_source

    def supports(self, scope: Scope) -> bool:
        return scope == self.scope

    def supports_resume(self, scope: Scope) -> bool:
        return (
            self.supports(scope)
            and self.resume_configuration_loader is not None
            and self.resume_manifest_supported is True
        )

    def owns_resume(self, scope: Scope, source: str, binding: ResumeBinding) -> bool:
        return (
            self.supports_resume(scope)
            and self.process is not None
            and self._current_source == source
            and self.resume_camera_id == binding.cameraId
            and self._owned_generation is not None
            and self._owned_generation.get("workerGenerationId") == binding.workerGenerationId
            and self._owned_generation.get("streamSessionId") == binding.streamSessionId
        )

    def supports_stopped_owner_resume(self, scope: Scope) -> bool:
        return (
            self.supports_resume(scope)
            and self.capture_lease_directory is not None
            and self.capture_lease_resource_id is not None
        )

    def can_resume_stopped_owner(self, scope: Scope, source: str, binding: ResumeBinding) -> bool:
        return (
            self.supports_stopped_owner_resume(scope)
            and not self._reconciliation_required
            and self.process is None
            and self._current_source is None
            and self._owned_generation is None
            and isinstance(source, str)
            and bool(source)
            and self.resume_camera_id == binding.cameraId
        )

    async def verify_unowned_capture_released(self, scope: Scope) -> None:
        if (
            not self.supports_stopped_owner_resume(scope)
            or self._reconciliation_required
            or self.process is not None
            or self._current_source is not None
            or self._owned_generation is not None
            or self.capture_lease_directory is None
            or self.capture_lease_resource_id is None
        ):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        lease = ConfigurationProcessLease(
            self.capture_lease_directory, self.capture_lease_resource_id
        )
        try:
            lease.acquire()
        except (OSError, RuntimeError, ValueError):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        finally:
            lease.release()

    def _write_generation_metadata(self, worker_generation_id: object) -> None:
        """Persist non-secret active generation ID atomically for stopped-owner recovery."""
        if not isinstance(worker_generation_id, str):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        try:
            normalized = str(UUID(worker_generation_id))
        except (ValueError, TypeError, AttributeError):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        path = self.generation_file
        if str(path).startswith("\\\\"):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        parent = path.parent
        try:
            if (
                not parent.is_dir()
                or is_link_like(parent)
                or (os.path.lexists(path) and (is_link_like(path) or path.is_dir()))
            ):
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            resolved_parent = parent.resolve(strict=True)
            resolved_target = resolved_parent / path.name
            if os.path.lexists(resolved_target) and (
                is_link_like(resolved_target) or resolved_target.is_dir()
            ):
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        except (OSError, DatasetIntegrityError):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None

        data = json.dumps({"workerGenerationId": normalized}, separators=(",", ":")).encode("utf-8")
        tmp = resolved_target.with_name(
            f".{resolved_target.name}.{os.getpid()}.{uuid4().hex[:12]}.tmp"
        )
        fd: int | None = None
        created = False
        try:
            fd = os.open(
                tmp,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
            created = True
            written = 0
            while written < len(data):
                count = os.write(fd, data[written:])
                if count <= 0:
                    raise OSError("Incomplete generation metadata write")
                written += count
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp, resolved_target)
        except OSError:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            if created:
                with suppress(OSError):
                    tmp.unlink(missing_ok=True)
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        except BaseException:
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            if created:
                with suppress(OSError):
                    tmp.unlink(missing_ok=True)
            raise

    async def generation_id(self, scope: Scope) -> str | None:
        if not self.supports(scope):
            raise SourceActivationError()
        if self._reconciliation_required:
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        if self._owned_generation is not None:
            return str(self._owned_generation["workerGenerationId"])
        try:
            if os.path.lexists(self.generation_file) and (
                is_link_like(self.generation_file) or self.generation_file.is_dir()
            ):
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            with self.generation_file.open("rb") as handle:
                raw = handle.read(_MAX_GENERATION_FILE_BYTES + 1)
        except FileNotFoundError:
            if (
                self.allow_unowned_start
                and self.process is None
                and not self.generation_file.is_symlink()
            ):
                return None
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        except (OSError, DatasetIntegrityError):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        if len(raw) > _MAX_GENERATION_FILE_BYTES:
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        try:
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
        if (
            not isinstance(data, dict)
            or set(data) != {"workerGenerationId"}
            or not isinstance(data["workerGenerationId"], str)
            or not data["workerGenerationId"]
        ):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        return data["workerGenerationId"]

    async def stop(self, scope: Scope) -> None:
        if not self.supports(scope):
            raise SourceActivationError()
        if self._reconciliation_required:
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        self.stop_file.write_text("stop", encoding="ascii")
        if self.process is not None:
            try:
                async with asyncio.timeout(15):
                    await self.process.wait()
            except TimeoutError:
                # Forced termination is containment, not proof of a cooperative
                # capture/lease drain. Preserve ownership and block replacement,
                # including if termination or waiting is cancelled or fails.
                self._reconciliation_required = True
                self.process.kill()
                await self.process.wait()
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED") from None
            self.process = None
        self._current_source = None
        self._owned_generation = None
        # The operator must stop any pre-existing reader before this driver starts.
        # A fresh host process cannot infer ownership from a PID or port alone.

    async def start(self, scope: Scope, source: str, generation: Mapping[str, str]) -> None:
        await self._start(scope, source, generation)

    async def start_resume(
        self, scope: Scope, source: str, generation: Mapping[str, str], manifest: ResumeManifest
    ) -> None:
        if self._reconciliation_required:
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        if not self.supports_resume(scope) or not isinstance(manifest, ResumeManifest):
            raise SourceActivationError("CONFIGURATION_RESUME_UNAVAILABLE")
        manifest = ResumeManifest.model_validate(manifest.model_dump())
        generation = dict(generation)
        if (
            manifest.generationBinding.cameraId != self.resume_camera_id
            or manifest.generationBinding.workerGenerationId != generation.get("workerGenerationId")
            or manifest.generationBinding.streamSessionId != generation.get("streamSessionId")
        ):
            raise SourceActivationError()
        assert self.resume_configuration_loader is not None
        async with asyncio.timeout(10):
            loaded = await self.resume_configuration_loader(manifest, generation)
        if not isinstance(loaded, ResumeConfiguration):
            raise SourceActivationError("EXPECTED_CONFIGURATION_UNAVAILABLE")
        manifest.require_configuration(ResumeConfiguration.model_validate(loaded.model_dump()))
        await self._start(scope, source, generation, manifest)

    async def _start(
        self,
        scope: Scope,
        source: str,
        generation: Mapping[str, str],
        manifest: ResumeManifest | None = None,
    ) -> None:
        if self._reconciliation_required:
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        if not self.supports(scope) or self.process is not None:
            raise SourceActivationError()
        self.stop_file.unlink(missing_ok=True)
        if self.capture_lease_directory is not None and self.capture_lease_resource_id is not None:
            self._write_generation_metadata(generation.get("workerGenerationId"))
        env = dict(self.environment)
        env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = source
        env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"] = json.dumps(dict(generation))
        # A fixed reviewed boundary; caller environment cannot smuggle a stale manifest.
        env.pop("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", None)
        if manifest is not None:
            env["SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION"] = manifest.model_dump_json()
        self.process = await asyncio.create_subprocess_exec(
            *(self.resume_command if manifest is not None else self.command),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=int(getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)),
        )
        self._current_source = source
        self._owned_generation = dict(generation)

    async def is_running(self, scope: Scope, generation_id: str) -> bool:
        if (
            self._reconciliation_required
            or not self.supports(scope)
            or self.process is None
            or self._owned_generation is None
            or self._owned_generation.get("workerGenerationId") != generation_id
        ):
            return False
        for _ in range(70):
            if self.process.returncode is not None:
                return False
            if await self.ready(generation_id):
                return self.process is not None and self.process.returncode is None
            await asyncio.sleep(0.4)
        return False
