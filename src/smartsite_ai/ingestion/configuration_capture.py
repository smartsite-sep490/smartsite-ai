"""Explicit capture-only ownership. API observers cannot start or restart this worker."""

import asyncio
import contextlib
import copy
import math
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from smartsite_ai.configuration_image import (
    ConfigurationImagePayload,
    ConfigurationJpeg,
    SourceKind,
    configuration_payload,
    encode_configuration_image,
)
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.ingestion.source import FrameSource


class ConfigurationSourceLease:
    """One explicitly shared source owner, not a global or cross-process registry.

    The runtime must bind one lease to one physical resource before activation;
    alias/cross-process exclusion remains a deployment precondition. This lease
    stays claimed until all source I/O, encoding and close finish successfully.
    """

    def __init__(
        self, source: FrameSource, camera_external_id: str, *, source_kind: SourceKind
    ) -> None:
        if source_kind not in ("LIVE", "REPLAY"):
            raise ValueError("Configuration source kind must be explicit")
        if (
            not camera_external_id.strip()
            or len(camera_external_id) > 128
            or any(unicodedata.category(c) in {"Cc", "Cs"} for c in camera_external_id)
        ):
            raise ValueError("Configuration camera identity is invalid")
        self._source = source
        self._camera_external_id = camera_external_id
        self._source_kind = source_kind
        self._stream_id = source.source_id
        self._owner: object | None = None

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def camera_external_id(self) -> str:
        return self._camera_external_id

    @property
    def source_kind(self) -> SourceKind:
        return self._source_kind

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def claim(self, owner: object) -> None:
        if self._owner is not None or self.source.is_connected:
            raise RuntimeError("Configuration source ownership is unavailable")
        self._owner = owner

    def release(self, owner: object) -> None:
        if self._owner is not owner:
            raise RuntimeError("Configuration source owner does not match")
        self._owner = None


class ConfigurationCapture:
    """Single-use lifecycle, one pending frame, one encoder, no implicit recovery.

    A bounded stop response is NOT proof that native I/O stopped. Never cancel
    a native read then close its handle underneath it: pending I/O must drain.
    Until then pixels disappear and the lease stays held. Failed close retains
    ownership permanently for explicit operator recovery, rather than reopening.
    """

    def __init__(
        self,
        lease: ConfigurationSourceLease,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        encoder: Callable[[FrameEnvelope], ConfigurationJpeg] = encode_configuration_image,
        stop_timeout: float = 2.0,
    ) -> None:
        if not math.isfinite(stop_timeout) or not 0 < stop_timeout <= 10:
            raise ValueError("Configuration stop budget must be bounded")
        self._lease = lease
        self._clock = clock
        self._encoder = encoder
        self._stop_timeout = stop_timeout
        self._state: Literal["IDLE", "RUNNING", "STOPPING", "STOPPED", "FAILED"] = "IDLE"
        self._reason: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._pending: FrameEnvelope | None = None
        self._latest: ConfigurationImagePayload | None = None
        self._latest_time: datetime | None = None
        self._session: UUID | None = None
        self._sequence = -1
        self._released = False
        self._generation = 0

    @property
    def camera_external_id(self) -> str:
        return self._lease.camera_external_id

    @property
    def state(self) -> str:
        return self._state

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def pending_frames(self) -> int:
        return int(self._pending is not None)

    def _fresh(self, captured_at: datetime) -> bool:
        return -1 <= (self._clock() - captured_at).total_seconds() <= 2

    def latest(self) -> ConfigurationImagePayload | None:
        """Observe only, without I/O, encoding, lifecycle changes or token state."""
        if (
            self._state != "RUNNING"
            or self._latest_time is None
            or not self._fresh(self._latest_time)
        ):
            return None
        return copy.deepcopy(self._latest)

    async def start(self) -> None:
        if self._task is not None and not self._task.done() and not self._stop.is_set():
            return
        if self._state != "IDLE":
            raise RuntimeError("Configuration capture lifecycle has ended")
        self._lease.claim(self)
        self._state = "RUNNING"
        self._task = asyncio.create_task(self._run(), name="configuration-capture-owner")

    def _clear(self) -> None:
        self._generation += 1
        self._latest = self._latest_time = self._pending = None

    def _fail(self, reason: str) -> None:
        self._reason = reason
        self._state = "FAILED"
        self._stop.set()
        self._clear()
        self._ready.set()

    def _valid_identity(self, frame: FrameEnvelope) -> bool:
        if (
            frame.camera_external_id != self.camera_external_id
            or frame.stream_id != self._lease.stream_id
            or (self._session is None and frame.sequence_number != 0)
            or (self._session is not None and frame.session_id != self._session)
            or frame.sequence_number <= self._sequence
        ):
            return False
        self._session, self._sequence = frame.session_id, frame.sequence_number
        return True

    async def _encode_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await self._ready.wait()
                self._ready.clear()
                frame, self._pending = self._pending, None
                if frame is None or self._stop.is_set() or not self._fresh(frame.captured_at):
                    continue
                generation = self._generation
                jpeg = await asyncio.to_thread(self._encoder, frame)
                if (
                    not self._stop.is_set()
                    and generation == self._generation
                    and self._fresh(frame.captured_at)
                ):
                    self._latest = configuration_payload(frame, self._lease.source_kind, jpeg)
                    self._latest_time = frame.captured_at
                # At most two encodes/second. While waiting, only the newest raw
                # frame is retained; no viewer request can increase this work.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=0.5)
        except Exception:
            self._fail("image_encoding_failed")

    async def _run(self) -> None:
        encoder_task: asyncio.Task[None] | None = None
        source_cancelled = False
        try:
            await self._lease.source.connect()
            if self._stop.is_set():
                return
            encoder_task = asyncio.create_task(
                self._encode_loop(), name="configuration-jpeg-encoder"
            )
            while not self._stop.is_set():
                frame = await self._lease.source.read_frame()
                if self._stop.is_set():
                    break
                if frame is None:
                    self._reason = "source_ended"
                    break
                if not self._valid_identity(frame):
                    self._fail("frame_identity_invalid")
                    break
                if self._fresh(frame.captured_at):
                    self._pending = frame
                    self._ready.set()
                else:
                    self._clear()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # An unexpected external cancellation can abandon a native thread.
            # Fail closed and retain ownership; never race close against that I/O.
            source_cancelled = True
            self._fail("capture_cancelled")
            raise
        except Exception:
            self._fail("source_failed")
        finally:
            self._stop.set()
            self._clear()
            self._ready.set()
            if encoder_task is not None:
                await encoder_task
            # Unexpected source cancellation has no native-release proof. Only
            # explicit runtime recovery may reclaim that still-poisoned lease.
            if not source_cancelled:
                await self._close_source()

    async def _close_source(self) -> None:
        try:
            await self._lease.source.close()
        except asyncio.CancelledError:
            self._fail("capture_cancelled")
            raise
        except Exception:
            self._fail("source_release_failed")
        else:
            self._lease.release(self)
            self._released = True
            if self._state != "FAILED":
                self._state = "STOPPED"

    async def stop(self) -> bool:
        """Clear immediately, wait a bounded time; false means ownership unresolved.

        Cancelling a stop waiter never cancels the source/encoder/cleanup tasks.
        Subsequent stop calls may observe completion but never repeat close.
        """
        self._stop.set()
        self._clear()
        self._ready.set()
        if self._state not in ("FAILED", "STOPPED"):
            self._state = "STOPPING"
        if self._task is None:
            self._state = "STOPPED"
            return True
        await asyncio.wait({self._task}, timeout=self._stop_timeout)
        if self._task.done() and not self._task.cancelled():
            self._task.result()
        return self._released
