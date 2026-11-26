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
    async def generation_id(self, scope: Scope) -> str | None: ...
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


@runtime_checkable
class StoppedOwnerResumeDriver(Protocol):
    """Opt-in interface verifying unowned stopped-owner recovery against an OS capture lease."""

    def supports_stopped_owner_resume(self, scope: Scope) -> bool: ...
    def can_resume_stopped_owner(
        self, scope: Scope, source: str, binding: ResumeBinding
    ) -> bool: ...
    async def verify_unowned_capture_released(self, scope: Scope) -> None: ...


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
            history = self._resume_history(old) if old else {}
            if command in history:
                raise SourceActivationError("SOURCE_RESUME_SUPERSEDED")
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
            if generation is None:
                if old is not None or revision != 0:
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            else:
                UUID(generation)
            row = {
                "state": "STOPPING",
                "commandId": command,
                "requestHash": fingerprint,
                "sourceRevision": revision,
                "sourceKind": tested[2],
                "source": tested[1],
                "currentGenerationId": generation,
                "resumeHistory": history,
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
            row.pop("resumeHistory", None)
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

    @staticmethod
    def _resume_history(row: dict[str, object]) -> dict[str, object]:
        saved = row.get("resumeHistory", {})
        if not isinstance(saved, dict):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        history = dict(saved)
        # Support the prior single-command encrypted registry without discarding its command.
        if row.get("operation") == "RESUME":
            request = ResumePrepare.model_validate(row["resumeRequest"])
            history.setdefault(request.commandId, request.model_dump())
        return history

    async def resume_status(self, scope: Scope, command: str) -> dict[str, object]:
        command = str(UUID(command))
        async with self.lock:
            self._resume_driver(scope)
            row = self.registry.read(scope)
            if not row:
                raise SourceActivationError("SOURCE_RESUME_NOT_FOUND")
            if row.get("operation") != "RESUME" and row.get("state") != "RUNNING":
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            history = self._resume_history(row)
            if command not in history:
                raise SourceActivationError("SOURCE_RESUME_NOT_FOUND")
            if row.get("commandId") != command or row.get("operation") != "RESUME":
                raise SourceActivationError("SOURCE_RESUME_SUPERSEDED")
            request = ResumePrepare.model_validate(history[command])
            if request.expectedRevision != row.get("sourceRevision"):
                raise SourceActivationError("SOURCE_RESUME_SUPERSEDED")
            result: dict[str, object] = {
                "commandId": command,
                "sourceRevision": request.expectedRevision,
                "sourceKind": "LIVE",
                "state": row["state"],
                "generationBinding": request.generationBinding.model_dump(),
                "configuration": request.configuration.model_dump(),
            }
            if row["state"] == "RUNNING":
                if not self._owns_running_resume(scope, row) or not await self.driver.is_running(
                    scope, str(row["currentGenerationId"])
                ):
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                result["newGenerationBinding"] = ResumeManifest.model_validate(
                    row["resumeManifest"]
                ).generationBinding.model_dump()
            return result

    async def _verify_stopped_owner(
        self,
        driver: ResumeDriver,
        scope: Scope,
        source: object,
        binding: ResumeBinding,
    ) -> None:
        if (
            not isinstance(source, str)
            or not source
            or not isinstance(driver, StoppedOwnerResumeDriver)
            or not driver.supports_stopped_owner_resume(scope)
            or not driver.can_resume_stopped_owner(scope, source, binding)
        ):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        persisted_generation = await self.driver.generation_id(scope)
        if not isinstance(persisted_generation, str) or persisted_generation != (
            binding.workerGenerationId
        ):
            raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
        await driver.verify_unowned_capture_released(scope)

    async def prepare_resume(self, scope: Scope, request: ResumePrepare) -> dict[str, object]:
        # Revalidate even callers using model_copy/model_construct; freeze before awaiting.
        request = ResumePrepare.model_validate(request.model_dump())
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        async with self.lock:
            driver = self._resume_driver(scope)
            row = self.registry.read(scope)
            history = self._resume_history(row) if row else {}
            if request.commandId in history and row.get("commandId") != request.commandId:
                raise SourceActivationError("SOURCE_RESUME_SUPERSEDED")
            binding = request.generationBinding
            if row and row.get("commandId") == request.commandId:
                if row.get("operation") != "RESUME" or row.get("requestHash") != fingerprint:
                    raise SourceActivationError()
                if row.get("state") not in {"PREPARED", "RUNNING"}:
                    raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                if row["state"] == "RUNNING":
                    if not self._owns_running_resume(
                        scope, row
                    ) or not await self.driver.is_running(scope, str(row["currentGenerationId"])):
                        raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                    return self._public(row)
                source = row.get("source")
                if (
                    (not isinstance(source, str) or not driver.owns_resume(scope, source, binding))
                    and isinstance(driver, StoppedOwnerResumeDriver)
                    and driver.supports_stopped_owner_resume(scope)
                ):
                    prior = row.get("generation")
                    if (
                        row.get("sourceKind") != "LIVE"
                        or type(row.get("sourceRevision")) is not int
                        or row.get("sourceRevision") != request.expectedRevision
                        or row.get("currentGenerationId") != binding.workerGenerationId
                        or not isinstance(prior, dict)
                        or prior.get("workerGenerationId") != binding.workerGenerationId
                        or prior.get("streamSessionId") != binding.streamSessionId
                    ):
                        raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
                    await self._verify_stopped_owner(driver, scope, source, binding)
                return self._public(row)
            prior = row.get("generation") if row else None
            if (
                not row
                or row.get("state") != "RUNNING"
                or row.get("sourceKind") != "LIVE"
                or type(row.get("sourceRevision")) is not int
                or row.get("sourceRevision") != request.expectedRevision
                or row.get("currentGenerationId") != binding.workerGenerationId
                or not isinstance(prior, dict)
                or prior.get("workerGenerationId") != binding.workerGenerationId
                or prior.get("streamSessionId") != binding.streamSessionId
                or not isinstance(row.get("source"), str)
            ):
                raise SourceActivationError("SOURCE_RECONCILIATION_REQUIRED")
            if not driver.owns_resume(scope, row["source"], binding):
                await self._verify_stopped_owner(driver, scope, row["source"], binding)
            row.update(
                state="STOPPING",
                operation="RESUME",
                commandId=request.commandId,
                requestHash=fingerprint,
                resumeRequest=request.model_dump(),
                resumeHistory={**history, request.commandId: request.model_dump()},
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
