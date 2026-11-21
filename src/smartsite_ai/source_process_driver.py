"""One explicitly configured camera process, with no adoption of unknown owners."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from uuid import UUID

from smartsite_ai.source_activation import Scope, SourceActivationError
from smartsite_ai.source_resume_models import ResumeBinding, ResumeConfiguration, ResumeManifest


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
        resume_camera_id: str | None = None,
        resume_manifest_supported: bool = False,
        resume_configuration_loader: Callable[
            [ResumeManifest, Mapping[str, str]], Awaitable[ResumeConfiguration]
        ]
        | None = None,
    ) -> None:
        if (
            not command
            or not Path(command[0]).is_absolute()
            or not stop_file.is_absolute()
            or not generation_file.is_absolute()
        ):
            raise ValueError("Reviewed absolute runtime paths required")
        forbidden = {"DATABASE_URL", "TEST_DATABASE_URL", "PGPASSWORD", "PGUSER"}
        if forbidden.intersection(k.upper() for k in environment):
            raise ValueError("Worker must not receive database credentials")
        self.scope = scope
        self.command = command
        self.stop_file = stop_file
        self.generation_file = generation_file
        self.environment = dict(environment)
        self.ready = ready
        if (resume_camera_id is None) != (resume_configuration_loader is None):
            raise ValueError("Resume requires a Camera binding and guarded configuration loader")
        self.resume_camera_id = str(UUID(resume_camera_id)) if resume_camera_id else None
        self.resume_manifest_supported = resume_manifest_supported
        # Opt-in asserts that the reviewed command also consumes the fixed manifest env below
        # and verifies it before model/capture startup; an arbitrary legacy command is insufficient.
        self.resume_configuration_loader = resume_configuration_loader
        self.process: asyncio.subprocess.Process | None = None
        self._current_source: str | None = None
        self._owned_generation: dict[str, str] | None = None

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

    async def generation_id(self, scope: Scope) -> str:
        if not self.supports(scope):
            raise SourceActivationError()
        data = json.loads(self.generation_file.read_bytes())
        return str(data["workerGenerationId"])

    async def stop(self, scope: Scope) -> None:
        if not self.supports(scope):
            raise SourceActivationError()
        self.stop_file.write_text("stop", encoding="ascii")
        if self.process is not None:
            try:
                async with asyncio.timeout(15):
                    await self.process.wait()
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
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
        if not self.supports(scope) or self.process is not None:
            raise SourceActivationError()
        self.stop_file.unlink(missing_ok=True)
        env = dict(self.environment)
        env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = source
        env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"] = json.dumps(dict(generation))
        # A fixed reviewed boundary; caller environment cannot smuggle a stale manifest.
        env.pop("SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION", None)
        if manifest is not None:
            env["SMARTSITE_AI_SOURCE_RUNTIME_EXPECTED_CONFIGURATION"] = manifest.model_dump_json()
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
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
            not self.supports(scope)
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
