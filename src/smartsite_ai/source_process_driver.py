"""One explicitly configured camera process, with no adoption of unknown owners."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from smartsite_ai.source_activation import Scope, SourceActivationError


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
        self.process: asyncio.subprocess.Process | None = None
        self.current_source: str | None = None

    def supports(self, scope: Scope) -> bool:
        return scope == self.scope

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
        self.current_source = None
        # The operator must stop any pre-existing reader before this driver starts.
        # A fresh host process cannot infer ownership from a PID or port alone.

    async def start(self, scope: Scope, source: str, generation: Mapping[str, str]) -> None:
        if not self.supports(scope) or self.process is not None:
            raise SourceActivationError()
        self.stop_file.unlink(missing_ok=True)
        env = dict(self.environment)
        env["SMARTSITE_AI_SOURCE_RUNTIME_SOURCE"] = source
        env["SMARTSITE_AI_SOURCE_RUNTIME_GENERATION"] = json.dumps(dict(generation))
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=int(getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)),
        )
        self.current_source = source

    async def is_running(self, scope: Scope, generation_id: str) -> bool:
        if not self.supports(scope) or self.process is None:
            return False
        for _ in range(70):
            if self.process.returncode is not None:
                return False
            if await self.ready(generation_id):
                return True
            await asyncio.sleep(0.4)
        return False
