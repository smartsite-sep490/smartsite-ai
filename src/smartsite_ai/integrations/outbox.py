"""Crash-tolerant SQLite outbox for Backend observation delivery."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from uuid import UUID

import httpx

from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.integrations.backend_client import (
    BackendClient,
    BoundDeliveryCredentialUnavailable,
)
from smartsite_ai.integrations.worker_generation import WorkerGenerationCredential


class OutboxState(StrEnum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    TERMINAL = "TERMINAL"


class OutboxEnqueueOutcome(StrEnum):
    ENQUEUED = "ENQUEUED"
    IDEMPOTENT = "IDEMPOTENT"


class OutboxConflict(ValueError):
    """One event ID was reused with different canonical event content."""


class OutboxDeliveryBlocked(RuntimeError):
    """Delivery cannot proceed until an operator fixes Backend integration."""


@dataclass(frozen=True, slots=True)
class PendingOutboxEvent:
    event: TechnicalObservationEvent
    payload_hash: str
    attempt_count: int
    delivery_context: EventDeliveryContext | None = None


@dataclass(frozen=True, slots=True)
class OutboxCounts:
    pending: int
    delivered: int
    terminal: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("outbox clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


class SqliteEventOutbox:
    """Persist events before delivery and retain their terminal outcome.

    SQLite operations run in a worker thread and are serialized by an asyncio
    lock. A successful Backend response may be followed by a process crash
    before ``mark_delivered`` commits; the event then remains pending and is
    safely retried with the same event ID.
    """

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not path.is_absolute():
            raise ValueError("outbox path must be absolute")
        if path.exists() and path.is_dir():
            raise ValueError("outbox path must identify a SQLite file")
        if not path.parent.is_dir():
            raise ValueError("outbox parent directory must already exist")
        self._path = path
        self._clock = clock
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def path(self) -> Path:
        return self._path

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize_sync(self) -> None:
        # Concurrent fresh processes can race while enabling WAL before busy_timeout applies.
        for attempt in range(4):
            try:
                self._initialize_once_sync()
                return
            except sqlite3.OperationalError as error:
                if attempt == 3 or getattr(error, "sqlite_errorcode", None) not in (
                    sqlite3.SQLITE_BUSY,
                    sqlite3.SQLITE_LOCKED,
                ):
                    raise
                time.sleep(0.01 * 2**attempt)

    def _initialize_once_sync(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS worker_session_claim (
                    generation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    camera_id TEXT NOT NULL,
                    claimed_at REAL NOT NULL
                ) STRICT"""
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_outbox (
                    event_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL CHECK(length(payload_hash) = 64),
                    state TEXT NOT NULL CHECK(state IN ('PENDING', 'DELIVERED', 'TERMINAL')),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
                    next_attempt_at REAL NOT NULL,
                    last_error_class TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delivered_at REAL
                ) STRICT
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS observation_outbox_ready_idx
                ON observation_outbox(state, next_attempt_at, created_at)
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(observation_outbox)")
            }
            if "delivery_context_json" not in columns:
                connection.execute(
                    "ALTER TABLE observation_outbox ADD COLUMN delivery_context_json TEXT"
                )
            if "delivery_context_hash" not in columns:
                connection.execute(
                    "ALTER TABLE observation_outbox ADD COLUMN delivery_context_hash TEXT "
                    "CHECK(delivery_context_hash IS NULL OR length(delivery_context_hash)=64)"
                )

    async def claim_worker_session(
        self, camera_id: UUID, generation_id: UUID, session_id: UUID
    ) -> None:
        """Consume once before source opening, including failed startup attempts."""
        if not all(isinstance(value, UUID) for value in (camera_id, generation_id, session_id)):
            raise ValueError("Worker session claim requires UUID identities")
        await self.initialize()
        now = _require_aware_utc(self._clock()).timestamp()
        async with self._lock:
            await asyncio.to_thread(
                self._claim_worker_session_sync,
                str(camera_id),
                str(generation_id),
                str(session_id),
                now,
            )

    def _claim_worker_session_sync(
        self, camera_id: str, generation_id: str, session_id: str, now: float
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO worker_session_claim VALUES (?, ?, ?, ?)",
                    (generation_id, session_id, camera_id, now),
                )
        except sqlite3.IntegrityError:
            raise RuntimeError("Worker generation or session is already consumed") from None

    async def enqueue(
        self,
        event: TechnicalObservationEvent,
        *,
        delivery_context: EventDeliveryContext | None = None,
    ) -> OutboxEnqueueOutcome:
        context_json = self._context_json((event,), delivery_context)
        await self.initialize()
        payload = event.to_wire_dict()
        payload_hash = event.compute_payload_hash()
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        now = _require_aware_utc(self._clock()).timestamp()
        async with self._lock:
            if context_json is not None:
                return await asyncio.to_thread(
                    self._enqueue_sync,
                    event.event_id,
                    payload_json,
                    payload_hash,
                    now,
                    context_json,
                )
            return await asyncio.to_thread(
                self._enqueue_sync,
                event.event_id,
                payload_json,
                payload_hash,
                now,
            )

    def _enqueue_sync(
        self,
        event_id: str,
        payload_json: str,
        payload_hash: str,
        now: float,
        context_json: str | None = None,
    ) -> OutboxEnqueueOutcome:
        return self._enqueue_many_sync(
            ((event_id, payload_json, payload_hash),), now, context_json
        )[0]

    async def enqueue_many(
        self,
        events: tuple[TechnicalObservationEvent, ...],
        *,
        delivery_context: EventDeliveryContext | None = None,
    ) -> tuple[OutboxEnqueueOutcome, ...]:
        """Durably commit every part of a frame, or none of its new parts."""
        context_json = self._context_json(events, delivery_context)
        await self.initialize()
        rows = tuple(
            (
                event.event_id,
                json.dumps(
                    event.to_wire_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ),
                event.compute_payload_hash(),
            )
            for event in events
        )
        now = _require_aware_utc(self._clock()).timestamp()
        async with self._lock:
            if context_json is not None:
                return await asyncio.to_thread(self._enqueue_many_sync, rows, now, context_json)
            return await asyncio.to_thread(self._enqueue_many_sync, rows, now)

    @staticmethod
    def _context_json(
        events: tuple[TechnicalObservationEvent, ...], context: EventDeliveryContext | None
    ) -> str | None:
        if context is None:
            return None
        # Revalidate model_construct-created instances before persisting metadata.
        context = EventDeliveryContext.model_validate_json(context.model_dump_json(by_alias=True))
        if any(event.stream_session_id != context.stream_session_id for event in events):
            raise ValueError("delivery context session does not match event")
        return context.model_dump_json(by_alias=True)

    def _enqueue_many_sync(
        self, rows: tuple[tuple[str, str, str], ...], now: float, context_json: str | None = None
    ) -> tuple[OutboxEnqueueOutcome, ...]:
        outcomes: list[OutboxEnqueueOutcome] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for event_id, payload_json, payload_hash in rows:
                context_hash = (
                    None
                    if context_json is None
                    else EventDeliveryContext.model_validate_json(context_json).binding_hash(
                        payload_hash
                    )
                )
                existing = connection.execute(
                    "SELECT payload_hash,delivery_context_json,delivery_context_hash "
                    "FROM observation_outbox WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["payload_hash"] != payload_hash
                        or existing["delivery_context_json"] != context_json
                        or existing["delivery_context_hash"] != context_hash
                    ):
                        raise OutboxConflict(
                            "outbox event ID already exists with different canonical content"
                        )
                    outcomes.append(OutboxEnqueueOutcome.IDEMPOTENT)
                    continue
                connection.execute(
                    """
                    INSERT INTO observation_outbox (
                        event_id, payload_json, payload_hash, state,
                        attempt_count, next_attempt_at, created_at, updated_at,
                        delivery_context_json,delivery_context_hash
                    ) VALUES (?, ?, ?, 'PENDING', 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        payload_json,
                        payload_hash,
                        now,
                        now,
                        now,
                        context_json,
                        context_hash,
                    ),
                )
                outcomes.append(OutboxEnqueueOutcome.ENQUEUED)
        return tuple(outcomes)

    async def ready(self, *, limit: int = 32) -> tuple[PendingOutboxEvent, ...]:
        if limit < 1 or limit > 1024:
            raise ValueError("outbox ready limit must be between 1 and 1024")
        await self.initialize()
        now = _require_aware_utc(self._clock()).timestamp()
        async with self._lock:
            rows = await asyncio.to_thread(self._ready_sync, now, limit)
        pending: list[PendingOutboxEvent] = []
        for row in rows:
            try:
                event = TechnicalObservationEvent.model_validate_json(row["payload_json"])
            except Exception as exc:
                raise RuntimeError("outbox contains an invalid observation event") from exc
            if event.compute_payload_hash() != row["payload_hash"]:
                raise RuntimeError("outbox observation payload hash mismatch")
            context = None
            if row["delivery_context_json"] is not None or row["delivery_context_hash"] is not None:
                try:
                    context = EventDeliveryContext.model_validate_json(row["delivery_context_json"])
                    if (
                        context.binding_hash(row["payload_hash"]) != row["delivery_context_hash"]
                        or context.stream_session_id != event.stream_session_id
                    ):
                        raise ValueError("Invalid binding")
                except (ValueError, TypeError):
                    raise RuntimeError("outbox delivery context is invalid or corrupted") from None
            pending.append(
                PendingOutboxEvent(
                    event=event,
                    payload_hash=row["payload_hash"],
                    attempt_count=row["attempt_count"],
                    delivery_context=context,
                )
            )
        return tuple(pending)

    def _ready_sync(self, now: float, limit: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT payload_json, payload_hash, attempt_count,
                      delivery_context_json, delivery_context_hash
                    FROM observation_outbox
                    WHERE state = 'PENDING' AND next_attempt_at <= ?
                    ORDER BY created_at, event_id
                    LIMIT ?
                    """,
                    (now, limit),
                ).fetchall()
            )

    async def mark_delivered(self, event_id: str, payload_hash: str) -> None:
        await self._transition(event_id, payload_hash, OutboxState.DELIVERED, None, None)

    async def mark_terminal(
        self,
        event_id: str,
        payload_hash: str,
        error_class: str,
    ) -> None:
        await self._transition(event_id, payload_hash, OutboxState.TERMINAL, None, error_class)

    async def reschedule(
        self,
        event_id: str,
        payload_hash: str,
        *,
        delay: timedelta,
        error_class: str,
    ) -> None:
        if delay < timedelta(0):
            raise ValueError("outbox retry delay must not be negative")
        next_attempt = _require_aware_utc(self._clock()) + delay
        await self._transition(
            event_id,
            payload_hash,
            OutboxState.PENDING,
            next_attempt.timestamp(),
            error_class,
        )

    async def _transition(
        self,
        event_id: str,
        payload_hash: str,
        state: OutboxState,
        next_attempt_at: float | None,
        error_class: str | None,
    ) -> None:
        if error_class is not None and (not error_class or len(error_class) > 128):
            raise ValueError("outbox error class must contain 1 to 128 characters")
        await self.initialize()
        now = _require_aware_utc(self._clock()).timestamp()
        async with self._lock:
            await asyncio.to_thread(
                self._transition_sync,
                event_id,
                payload_hash,
                state,
                next_attempt_at,
                error_class,
                now,
            )

    def _transition_sync(
        self,
        event_id: str,
        payload_hash: str,
        state: OutboxState,
        next_attempt_at: float | None,
        error_class: str | None,
        now: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT payload_hash, state, next_attempt_at
                FROM observation_outbox
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError("outbox event does not exist")
            if row["payload_hash"] != payload_hash:
                raise OutboxConflict("outbox transition payload hash does not match stored content")
            if row["state"] == OutboxState.DELIVERED:
                if state == OutboxState.DELIVERED:
                    return
                raise RuntimeError("delivered outbox event cannot transition again")
            if row["state"] == OutboxState.TERMINAL:
                if state == OutboxState.TERMINAL:
                    return
                raise RuntimeError("terminal outbox event cannot transition again")

            delivered_at = now if state == OutboxState.DELIVERED else None
            effective_next_attempt = (
                next_attempt_at if next_attempt_at is not None else row["next_attempt_at"]
            )
            connection.execute(
                """
                UPDATE observation_outbox
                SET state = ?,
                    attempt_count = attempt_count + 1,
                    next_attempt_at = ?,
                    last_error_class = ?,
                    updated_at = ?,
                    delivered_at = ?
                WHERE event_id = ?
                """,
                (
                    state.value,
                    effective_next_attempt,
                    error_class,
                    now,
                    delivered_at,
                    event_id,
                ),
            )

    async def counts(self) -> OutboxCounts:
        await self.initialize()
        async with self._lock:
            values = await asyncio.to_thread(self._counts_sync)
        return OutboxCounts(
            pending=values.get(OutboxState.PENDING, 0),
            delivered=values.get(OutboxState.DELIVERED, 0),
            terminal=values.get(OutboxState.TERMINAL, 0),
        )

    def _counts_sync(self) -> dict[OutboxState, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM observation_outbox GROUP BY state"
            ).fetchall()
        return {OutboxState(row["state"]): row["count"] for row in rows}


class OutboxDispatcher:
    """Deliver ready events with bounded per-request retries and durable rescheduling."""

    def __init__(
        self,
        outbox: SqliteEventOutbox,
        backend: BackendClient,
        *,
        retry_base: timedelta = timedelta(seconds=1),
        retry_max: timedelta = timedelta(minutes=5),
        generation_credential: WorkerGenerationCredential | None = None,
    ) -> None:
        if retry_base <= timedelta(0):
            raise ValueError("dispatcher retry_base must be positive")
        if retry_max < retry_base:
            raise ValueError("dispatcher retry_max must be at least retry_base")
        self._outbox = outbox
        self._backend = backend
        self._retry_base = retry_base
        self._retry_max = retry_max
        self._generation_credential = generation_credential

    async def drain_once(self, *, limit: int = 32) -> int:
        processed = 0
        for pending in await self._outbox.ready(limit=limit):
            if pending.delivery_context is not None and self._generation_credential is None:
                await self._reschedule(pending, "BoundTransportUnavailable")
                raise OutboxDeliveryBlocked(
                    "Original delivery context requires authenticated bound transport"
                )
            try:
                if pending.delivery_context is None:
                    await self._backend.post_event(pending.event)
                else:
                    await self._backend.post_bound_event(
                        pending.event, pending.delivery_context, self._generation_credential
                    )
            except BoundDeliveryCredentialUnavailable:
                await self._reschedule(pending, "OriginalCredentialUnavailable")
                raise OutboxDeliveryBlocked(
                    "Original generation credential is unavailable"
                ) from None
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (400, 409, 413, 415, 422):
                    await self._outbox.mark_terminal(
                        pending.event.event_id,
                        pending.payload_hash,
                        f"HTTP_{status}",
                    )
                elif 400 <= status < 500 and status not in (408, 429):
                    # Authentication, authorization, route and other integration
                    # failures are operator-fixable. Preserve the event, then fail
                    # the supervised delivery loop so capture cannot keep producing
                    # an unbounded backlog with broken credentials/configuration.
                    await self._reschedule(pending, f"HTTP_{status}")
                    raise OutboxDeliveryBlocked(
                        f"Backend integration rejected outbox delivery with HTTP {status}"
                    ) from None
                else:
                    await self._reschedule(pending, f"HTTP_{status}")
            except (httpx.TransportError, TimeoutError) as exc:
                await self._reschedule(pending, type(exc).__name__)
            except ValueError:
                # A malformed 2xx acknowledgement is ambiguous: the Backend may
                # already have committed the event. Retry the same idempotent event
                # instead of losing it or inventing a new event ID.
                await self._reschedule(pending, "BackendResponseError")
            else:
                await self._outbox.mark_delivered(
                    pending.event.event_id,
                    pending.payload_hash,
                )
            processed += 1
        return processed

    async def _reschedule(self, pending: PendingOutboxEvent, error_class: str) -> None:
        multiplier = 2 ** min(pending.attempt_count, 30)
        delay_seconds = min(
            self._retry_base.total_seconds() * multiplier,
            self._retry_max.total_seconds(),
        )
        await self._outbox.reschedule(
            pending.event.event_id,
            pending.payload_hash,
            delay=timedelta(seconds=delay_seconds),
            error_class=error_class,
        )


__all__ = [
    "OutboxDeliveryBlocked",
    "OutboxConflict",
    "OutboxCounts",
    "OutboxDispatcher",
    "OutboxEnqueueOutcome",
    "OutboxState",
    "PendingOutboxEvent",
    "SqliteEventOutbox",
]
