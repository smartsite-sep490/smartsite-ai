"""Explicit generation-bound source changes; no process or storage IO on import."""

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from smartsite_ai.source_resume_models import ResumeBinding, ResumeManifest, ResumePrepare

Scope = tuple[str, str]


class SourceActivationError(ValueError):
    def __init__(self, reason: str = "SOURCE_ACTIVATION_CONFLICT") -> None:
        super().__init__(reason)


class SourceDriver(Protocol):
    def supports(self, scope: Scope) -> bool: ...
    async def generation_id(self, scope: Scope) -> str: ...
    async def stop(self, scope: Scope) -> None: ...
    async def start(self, scope: Scope, source: str, generation: Mapping[str, str]) -> None: ...
    async def is_running(self, scope: Scope, generation_id: str) -> bool: ...


@runtime_checkable
class ResumeDriver(Protocol):
    def supports_resume(self, scope: Scope) -> bool: ...
    def owns_resume(self, scope: Scope, source: str, binding: ResumeBinding) -> bool: ...
    async def start_resume(
        self, scope: Scope, source: str, generation: Mapping[str, str], manifest: ResumeManifest
    ) -> None: ...


class SourceRegistry:
    """One coordinator owns this encrypted file; no cross-process writer sharing."""

    def __init__(self, path: Path, key: SecretStr) -> None:
        self.path = path
        self.cipher = Fernet(key.get_secret_value().encode("ascii"))

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            if self.path.is_symlink():
                raise SourceActivationError("SOURCE_STORAGE_UNAVAILABLE")
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}
        if len(raw) > 1_000_000:
            raise SourceActivationError("SOURCE_STORAGE_UNAVAILABLE")
        try:
            value = json.loads(self.cipher.decrypt(raw))
            if not isinstance(value, dict) or any(not isinstance(v, dict) for v in value.values()):
                raise ValueError()
            return value
        except (InvalidToken, ValueError, TypeError):
            raise SourceActivationError("SOURCE_STORAGE_UNAVAILABLE") from None

    @staticmethod
    def _key(scope: Scope) -> str:
        return json.dumps(scope, separators=(",", ":"))

    def read(self, scope: Scope) -> dict[str, object] | None:
        return self._read().get(self._key(scope))

    def write(self, scope: Scope, value: dict[str, object]) -> None:
        state = self._read()
        state[self._key(scope)] = value
        encrypted = self.cipher.encrypt(json.dumps(state, separators=(",", ":")).encode())
        if len(encrypted) > 1_000_000:
            raise SourceActivationError("SOURCE_STORAGE_UNAVAILABLE")
        temporary = self.path.with_name(self.path.name + "." + str(uuid4()) + ".tmp")
        try:
            with temporary.open("xb") as output:
                output.write(encrypted)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class SourceActivator:
    def __init__(
        self,
        registry: SourceRegistry,
        driver: SourceDriver,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.registry = registry
        self.driver = driver
        self.clock = clock
        self.lock = asyncio.Lock()
        self.receipts: dict[str, tuple[Scope, str, str, datetime]] = {}

    def remember(self, scope: Scope, receipt: str, source: str, kind: str) -> None:
        now = self.clock()
        self.receipts = {k: v for k, v in self.receipts.items() if v[3] > now}
        if kind not in {"LIVE", "REPLAY"} or len(self.receipts) >= 128:
            raise SourceActivationError()
        UUID(receipt)
        self.receipts[receipt] = (scope, source, kind, now + timedelta(seconds=60))

    @staticmethod
    def _public(row: dict[str, object]) -> dict[str, object]:
        return {
            k: row[k]
            for k in ("state", "commandId", "sourceRevision", "sourceKind", "currentGenerationId")
            if k in row
        }

    async def prepare(
        self, scope: Scope, command: str, receipt: str, expected_revision: int
    ) -> dict[str, object]:
        UUID(command)
        async with self.lock:
            if not self.driver.supports(scope) or type(expected_revision) is not int:
                raise SourceActivationError()
            fingerprint = hashlib.sha256(
                json.dumps([receipt, expected_revision]).encode()
            ).hexdigest()
            old = self.registry.read(scope)
            if old and old.get("commandId") == command:
                if old.get("requestHash") != fingerprint:
                    raise SourceActivationError()
                if old.get("state") == "RUNNING" and not await self.driver.is_running(
                    scope, str(old["currentGenerationId"])
                ):
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                return self._public(old)
            if old and old.get("state") not in {"RUNNING", "STOPPED"}:
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            revision = old.get("sourceRevision", 0) if old else 0
            tested = self.receipts.get(receipt)
            if (
                revision != expected_revision
                or not tested
                or tested[0] != scope
                or tested[3] <= self.clock()
            ):
                raise SourceActivationError()
            if tested[2] != "LIVE":
                # The tracked operational pipeline must not relabel replay as live.
                raise SourceActivationError("REPLAY_ACTIVATION_UNAVAILABLE")
            generation = await self.driver.generation_id(scope)
            UUID(generation)
            row = {
                "state": "STOPPING",
                "commandId": command,
                "requestHash": fingerprint,
                "sourceRevision": revision,
                "sourceKind": tested[2],
                "source": tested[1],
                "currentGenerationId": generation,
            }
            self.registry.write(scope, row)
            # Persist uncertainty before side effects; cancellation requires reconciliation.
            await self.driver.stop(scope)
            row["state"] = "PREPARED"
            self.registry.write(scope, row)
            self.receipts.pop(receipt, None)
            return self._public(row)

    async def commit(
        self, scope: Scope, command: str, generation: Mapping[str, str]
    ) -> dict[str, object]:
        async with self.lock:
            row = self.registry.read(scope)
            if not row or row.get("commandId") != command or row.get("operation") == "RESUME":
                raise SourceActivationError()
            if row["state"] == "RUNNING":
                if row.get("currentGenerationId") != generation.get("workerGenerationId"):
                    raise SourceActivationError()
                if not await self.driver.is_running(scope, generation["workerGenerationId"]):
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                return self._public(row)
            if row["state"] != "PREPARED" or set(generation) != {
                "workerGenerationId",
                "streamSessionId",
                "secret",
                "expiresAt",
            }:
                raise SourceActivationError()
            UUID(generation["workerGenerationId"])
            UUID(generation["streamSessionId"])
            expiry = datetime.fromisoformat(generation["expiresAt"])
            if (
                expiry.tzinfo is None
                or expiry <= self.clock()
                or generation["workerGenerationId"] == row["currentGenerationId"]
                or not 32 <= len(generation["secret"]) <= 4096
            ):
                raise SourceActivationError()
            row.update(state="STARTING", generation=dict(generation))
            self.registry.write(scope, row)
            await self.driver.start(scope, str(row["source"]), generation)
            if not await self.driver.is_running(scope, generation["workerGenerationId"]):
                raise SourceActivationError("SOURCE_NOT_RUNNING")
            row.update(
                state="RUNNING",
                sourceRevision=int(row["sourceRevision"]) + 1,
                currentGenerationId=generation["workerGenerationId"],
            )
            self.registry.write(scope, row)
            return self._public(row)

    def supports_resume(self, scope: Scope) -> bool:
        return isinstance(self.driver, ResumeDriver) and self.driver.supports_resume(scope)

    def _resume_driver(self, scope: Scope) -> ResumeDriver:
        if not isinstance(self.driver, ResumeDriver) or not self.driver.supports_resume(scope):
            raise SourceActivationError("CONFIGURATION_RESUME_UNAVAILABLE")
        return self.driver

    def _owns_running_resume(self, scope: Scope, row: dict[str, object]) -> bool:
        manifest = ResumeManifest.model_validate(row["resumeManifest"])
        return self._resume_driver(scope).owns_resume(
            scope, str(row["source"]), manifest.generationBinding
        )

    async def prepare_resume(self, scope: Scope, request: ResumePrepare) -> dict[str, object]:
        # Revalidate even callers using model_copy/model_construct; freeze before awaiting.
        request = ResumePrepare.model_validate(request.model_dump())
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        async with self.lock:
            driver = self._resume_driver(scope)
            row = self.registry.read(scope)
            if row and row.get("commandId") == request.commandId:
                if row.get("operation") != "RESUME" or row.get("requestHash") != fingerprint:
                    raise SourceActivationError()
                if row.get("state") not in {"PREPARED", "RUNNING"}:
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                if row["state"] == "RUNNING" and (
                    not self._owns_running_resume(scope, row)
                    or not await self.driver.is_running(scope, str(row["currentGenerationId"]))
                ):
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                return self._public(row)
            binding = request.generationBinding
            prior = row.get("generation") if row else None
            if (
                not row
                or row.get("state") != "RUNNING"
                or row.get("sourceKind") != "LIVE"
                or row.get("sourceRevision") != request.expectedRevision
                or row.get("currentGenerationId") != binding.workerGenerationId
                or not isinstance(prior, dict)
                or prior.get("streamSessionId") != binding.streamSessionId
                or not isinstance(row.get("source"), str)
                or not driver.owns_resume(scope, row["source"], binding)
            ):
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            row.update(
                state="STOPPING",
                operation="RESUME",
                commandId=request.commandId,
                requestHash=fingerprint,
                resumeRequest=request.model_dump(),
            )
            self.registry.write(scope, row)
            await self.driver.stop(scope)
            row["state"] = "PREPARED"
            self.registry.write(scope, row)
            return self._public(row)

    async def commit_resume(
        self, scope: Scope, command: str, generation: Mapping[str, str]
    ) -> dict[str, object]:
        command = str(UUID(command))
        generation = dict(generation)
        async with self.lock:
            driver = self._resume_driver(scope)
            row = self.registry.read(scope)
            if not row or row.get("commandId") != command or row.get("operation") != "RESUME":
                raise SourceActivationError()
            if row["state"] == "RUNNING":
                if row.get("generation") != generation:
                    raise SourceActivationError()
                if not self._owns_running_resume(scope, row) or not await self.driver.is_running(
                    scope, generation["workerGenerationId"]
                ):
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                return self._public(row)
            if (
                row["state"] != "PREPARED"
                or set(generation)
                != {"workerGenerationId", "streamSessionId", "secret", "expiresAt"}
                or any(not isinstance(value, str) for value in generation.values())
            ):
                raise SourceActivationError()
            request = ResumePrepare.model_validate(row["resumeRequest"])
            binding = ResumeBinding(
                cameraId=request.generationBinding.cameraId,
                workerGenerationId=generation["workerGenerationId"],
                streamSessionId=generation["streamSessionId"],
            )
            expiry = datetime.fromisoformat(generation["expiresAt"])
            if (
                expiry.tzinfo is None
                or expiry <= self.clock()
                or binding.workerGenerationId == request.generationBinding.workerGenerationId
                or binding.streamSessionId == request.generationBinding.streamSessionId
                or not 32 <= len(generation["secret"]) <= 4096
            ):
                raise SourceActivationError()
            manifest = ResumeManifest(
                generationBinding=binding, configuration=request.configuration
            )
            row.update(
                state="STARTING", generation=generation, resumeManifest=manifest.model_dump()
            )
            self.registry.write(scope, row)
            await driver.start_resume(scope, str(row["source"]), generation, manifest)
            if not await self.driver.is_running(scope, binding.workerGenerationId):
                raise SourceActivationError("SOURCE_NOT_RUNNING")
            row.update(state="RUNNING", currentGenerationId=binding.workerGenerationId)
            self.registry.write(scope, row)
            return self._public(row)
