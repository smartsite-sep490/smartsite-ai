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
from typing import Any
from uuid import UUID

import httpx

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash
from smartsite_ai.domain.event_delivery_context import EventDeliveryContext
from smartsite_ai.domain.observations import TechnicalObservationEvent, validate_uuid_str
from smartsite_ai.domain.tracked_frame_result import TrackedFrameReferences, TrackedFrameResult
from smartsite_ai.domain.tracked_raw_receipt import TrackedRawReceipt
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


def _tracked_transient(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in (408, 429) or 500 <= error.response.status_code < 600
    return isinstance(error, (httpx.TransportError, TimeoutError))


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

            connection.execute(
                """CREATE TABLE IF NOT EXISTS tracked_frame_head (
                    scope_sha256 TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL,
                    result_json TEXT NOT NULL
                ) STRICT"""
            )

            connection.execute(
                """CREATE TABLE IF NOT EXISTS tracked_frame_pending (
                    camera_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    frame_id TEXT NOT NULL,
                    references_json TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    last_checked REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (camera_id, generation_id, session_id, frame_id)
                ) STRICT"""
            )

            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tracked_frame_pending)")
            }
            if "delivery_json" not in columns:
                connection.execute(
                    "ALTER TABLE tracked_frame_pending ADD COLUMN delivery_json TEXT"
                )

            if "raw_stored" not in columns:
                connection.execute(
                    "ALTER TABLE tracked_frame_pending ADD COLUMN raw_stored INTEGER"
                    " NOT NULL DEFAULT 0 CHECK(raw_stored IN (0, 1))"
                )
            for name, kind in (
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0)"),
                ("next_attempt_at", "REAL NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE tracked_frame_pending ADD COLUMN {name} {kind}"
                    )

    async def enqueue_tracked_parts(
        self,
        credential: WorkerGenerationCredential,
        context: EventDeliveryContext,
        parts: tuple[dict[str, Any], ...],
        *,
        expected_seal: TrackedFrameReferences | None,
        expires_at: datetime,
        max_pending: int,
        max_bytes: int,
    ) -> None:
        """Atomically retain draft wire bytes and their optional expected seal."""
        if (
            context.camera_id != credential.camera_id
            or context.worker_generation_id != credential.worker_generation_id
            or context.stream_session_id != credential.stream_session_id
        ):
            raise ValueError("Tracked frame context does not match the worker credential")
        if not 1 <= len(parts) <= 1024:
            raise ValueError("Tracked frame must contain a bounded complete set of parts")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Tracked frame size limit must be positive")

        partition = parts[0]["framePartition"]
        frame_id = validate_uuid_str(partition["frameId"], "frameId")
        sequence = partition["sequenceNumber"]
        coverage = parts[0]["continuity"]["coverage"]
        if (
            coverage["status"] not in ("COMPLETE", "INCOMPLETE")
            or (expected_seal is None) != (coverage["status"] == "INCOMPLETE")
            or (expected_seal is not None and expected_seal.sequence_number != sequence)
        ):
            raise ValueError("Tracked frame coverage and expected seal do not match")

        subjects: dict[str, tuple[str, int]] = {}
        retired: set[str] = set()
        event_ids: set[str] = set()
        wire = []
        total_bytes = 0
        for index, part in enumerate(parts):
            event_id = validate_uuid_str(part["eventId"], "eventId")
            if (
                part["schemaVersion"] != "1.5.0"
                or part["streamSessionId"] != context.stream_session_id
                or event_id in event_ids
                or part["framePartition"]
                != {
                    "frameId": frame_id,
                    "sequenceNumber": sequence,
                    "partIndex": index,
                    "partCount": len(parts),
                }
                or part["continuity"]["coverage"] != coverage
            ):
                raise ValueError("Tracked frame parts do not match their original frame or session")
            event_ids.add(event_id)
            for claim in part["continuity"]["subjects"]:
                person_index = claim["personObservationIndex"]
                observations = part["observations"]
                if (
                    not 0 <= person_index < len(observations)
                    or observations[person_index]["type"] != "PERSON"
                ):
                    raise ValueError("Tracked frame claim must reference an original PERSON")
                subjects.setdefault(claim["segmentId"], (event_id, person_index))
            retired.update(item["segmentId"] for item in part["continuity"]["retiredSegments"])

            body = json.dumps(
                {
                    "schemaVersion": "1.0.0",
                    "context": context.model_dump(by_alias=True),
                    "event": part,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            total_bytes += len(body.encode("utf-8"))
            if total_bytes > max_bytes:
                raise ValueError("Tracked frame exceeds its wire size limit")
            wire.append(body)

        if expected_seal is None:
            if subjects or retired:
                raise ValueError("Incomplete tracked frame cannot expect a seal")
            references_json = "null"
        else:
            expected_seal.require_match(expected_seal.scope_sha256, sequence, subjects, retired)
            references_json = expected_seal.model_dump_json(by_alias=True)

        await self._stage_tracked(
            credential,
            frame_id,
            references_json,
            expires_at=expires_at,
            max_pending=max_pending,
            delivery_json=json.dumps(wire, ensure_ascii=False),
        )

    async def ready_tracked_parts(
        self, credential: WorkerGenerationCredential, *, limit: int
    ) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
        """Return original envelope bytes only to their camera, generation and session."""
        now = _require_aware_utc(self._clock())
        credential.ensure_current(now)
        if type(limit) is not int or limit < 1:
            raise ValueError("Tracked frame lookup limit must be a positive integer")
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(
                self._ready_tracked_parts_sync, credential, now.timestamp(), limit
            )

    def _ready_tracked_parts_sync(
        self, credential: WorkerGenerationCredential, now: float, limit: int
    ) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT frame_id, delivery_json FROM tracked_frame_pending
                WHERE camera_id = ? AND generation_id = ? AND session_id = ?
                  AND expires_at > ? AND next_attempt_at <= ?
                  AND delivery_json IS NOT NULL AND raw_stored = 0
                ORDER BY rowid LIMIT ?""",
                (
                    credential.camera_id,
                    credential.worker_generation_id,
                    credential.stream_session_id,
                    now,
                    now,
                    limit,
                ),
            ).fetchall()
        return tuple(
            (
                row["frame_id"],
                tuple(body.encode("utf-8") for body in json.loads(row["delivery_json"])),
            )
            for row in rows
        )

    async def confirm_tracked_parts_stored(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        receipts: tuple[TrackedRawReceipt, ...],
    ) -> bool:
        """Promote only a full set of receipts obtained through authenticated transport."""
        credential.ensure_current(_require_aware_utc(self._clock()))
        frame_id = validate_uuid_str(frame_id, "frameId")
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(
                self._confirm_tracked_parts_stored_sync, credential, frame_id, receipts
            )

    def _confirm_tracked_parts_stored_sync(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        receipts: tuple[TrackedRawReceipt, ...],
    ) -> bool:
        key = (
            credential.camera_id,
            credential.worker_generation_id,
            credential.stream_session_id,
            frame_id,
        )
        where = """WHERE camera_id = ? AND generation_id = ? AND session_id = ?
            AND frame_id = ? AND expires_at > ? AND delivery_json IS NOT NULL"""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _require_aware_utc(self._clock()).timestamp()
            row = connection.execute(
                "SELECT delivery_json FROM tracked_frame_pending " + where, (*key, now)
            ).fetchone()
            if row is None:
                return False

            bodies = json.loads(row["delivery_json"])
            by_id = {receipt.event_id: receipt for receipt in receipts}
            if len(receipts) != len(bodies) or len(by_id) != len(receipts):
                raise ValueError("A distinct raw receipt is required for each tracked part")
            for body in bodies:
                envelope = json.loads(body)
                event = envelope["event"]
                event_id = event["eventId"]
                receipt = by_id.get(event_id)
                if receipt is None:
                    raise ValueError("Tracked part is missing its original raw receipt")
                payload_hash = compute_canonical_payload_hash(event)
                context = EventDeliveryContext.model_validate(envelope["context"])
                receipt.require_match(event_id, payload_hash, context.binding_hash(payload_hash))

            now = _require_aware_utc(self._clock()).timestamp()
            updated = connection.execute(
                """UPDATE tracked_frame_pending SET
                attempt_count = CASE WHEN raw_stored = 0 THEN 0 ELSE attempt_count END,
                next_attempt_at = CASE WHEN raw_stored = 0 THEN 0 ELSE next_attempt_at END,
                raw_stored = 1 """
                + where,
                (*key, now),
            )
            return updated.rowcount == 1

    async def defer_tracked_frame(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        *,
        retry_base: timedelta,
        retry_max: timedelta,
    ) -> None:
        """Persist bounded retry timing without changing original bytes or deadline."""
        if retry_base <= timedelta(0) or retry_max < retry_base:
            raise ValueError("Invalid tracked retry policy")
        now = _require_aware_utc(self._clock())
        credential.ensure_current(now)
        key = (
            credential.camera_id,
            credential.worker_generation_id,
            credential.stream_session_id,
            validate_uuid_str(frame_id, "frameId"),
        )
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(
                self._defer_tracked_frame_sync, key, now.timestamp(), retry_base, retry_max
            )

    def _defer_tracked_frame_sync(
        self, key: tuple[str, str, str, str], now: float, base: timedelta, cap: timedelta
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            where = """WHERE camera_id = ? AND generation_id = ? AND session_id = ?
                AND frame_id = ? AND expires_at > ? AND next_attempt_at <= ?"""
            args = (*key, now, now)
            row = connection.execute(
                "SELECT attempt_count FROM tracked_frame_pending " + where, args
            ).fetchone()
            if row is None:
                return
            delay = min(
                base.total_seconds() * 2 ** min(row["attempt_count"], 30), cap.total_seconds()
            )
            connection.execute(
                "UPDATE tracked_frame_pending SET attempt_count = attempt_count + 1, "
                "next_attempt_at = ? " + where,
                (now + delay, *args),
            )

    async def stage_tracked_frame(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        *,
        scope_sha256: str,
        sequence_number: str,
        subjects: dict[str, tuple[str, int]],
        retired_segment_ids: set[str],
        expires_at: datetime,
        max_pending: int,
    ) -> None:
        """Remember exact references before submitting a frame for admission."""
        frame_id = validate_uuid_str(frame_id, "frameId")
        references = TrackedFrameReferences.model_validate(
            {
                "scopeSha256": scope_sha256,
                "sequenceNumber": sequence_number,
                "subjects": [
                    {"segmentId": segment, "eventId": event, "personObservationIndex": index}
                    for segment, (event, index) in sorted(subjects.items())
                ],
                "retiredSegmentIds": sorted(retired_segment_ids),
            }
        )
        await self._stage_tracked(
            credential,
            frame_id,
            references.model_dump_json(by_alias=True),
            expires_at=expires_at,
            max_pending=max_pending,
        )

    async def _stage_tracked(
        self,
        credential: WorkerGenerationCredential,
        frame_id: str,
        references_json: str,
        *,
        expires_at: datetime,
        max_pending: int,
        delivery_json: str | None = None,
    ) -> None:
        now = _require_aware_utc(self._clock())
        expiry = _require_aware_utc(expires_at)
        credential.ensure_current(now)
        if not now < expiry <= datetime.fromisoformat(credential.expires_at):
            raise ValueError("Tracked frame lookup deadline is outside the credential lease")
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Tracked frame capacity must be a positive integer")
        source = (
            credential.camera_id,
            credential.worker_generation_id,
            credential.stream_session_id,
        )
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(
                self._stage_tracked_frame_sync,
                source,
                frame_id,
                references_json,
                now.timestamp(),
                expiry.timestamp(),
                max_pending,
                delivery_json,
            )

    def _stage_tracked_frame_sync(
        self,
        source: tuple[str, str, str],
        frame_id: str,
        references_json: str,
        now: float,
        expiry: float,
        max_pending: int,
        delivery_json: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT references_json, expires_at, delivery_json FROM tracked_frame_pending
                WHERE camera_id = ? AND generation_id = ? AND session_id = ? AND frame_id = ?""",
                (*source, frame_id),
            ).fetchone()
            if row is not None:
                if tuple(row) != (references_json, expiry, delivery_json):
                    raise OutboxConflict("Tracked frame lookup was staged with different content")
                return
            count = connection.execute(
                """SELECT count(*) FROM tracked_frame_pending
                WHERE camera_id = ? AND generation_id = ? AND session_id = ?
                  AND expires_at > ? AND (raw_stored = 0 OR references_json <> 'null')""",
                (*source, now),
            ).fetchone()[0]
            if count >= max_pending:
                raise OutboxConflict("Tracked frame lookup capacity reached")
            connection.execute(
                """INSERT INTO tracked_frame_pending
                (camera_id, generation_id, session_id, frame_id,
                  references_json, expires_at, delivery_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (*source, frame_id, references_json, expiry, delivery_json),
            )

    async def has_pending_tracked_frame(self, credential: WorkerGenerationCredential) -> bool:
        """Include expired unresolved work so source completion cannot look successful."""
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._has_pending_tracked_frame_sync, credential)

    def _has_pending_tracked_frame_sync(self, credential: WorkerGenerationCredential) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    """SELECT 1 FROM tracked_frame_pending
                WHERE camera_id = ? AND generation_id = ? AND session_id = ?
                  AND (raw_stored = 0 OR references_json <> 'null') LIMIT 1""",
                    (
                        credential.camera_id,
                        credential.worker_generation_id,
                        credential.stream_session_id,
                    ),
                ).fetchone()
                is not None
            )

    async def reconcile_tracked_frames(
        self,
        client: BackendClient,
        credential: WorkerGenerationCredential,
        *,
        limit: int,
        retry_base: timedelta | None = None,
        retry_max: timedelta | None = None,
    ) -> int:
        """Recheck a bounded batch with the original source credentials after restart."""
        now = _require_aware_utc(self._clock())
        credential.ensure_current(now)
        if type(limit) is not int or limit < 1:
            raise ValueError("Tracked frame lookup limit must be a positive integer")
        if (retry_base is None) != (retry_max is None) or (
            retry_base is not None and (retry_base <= timedelta(0) or retry_max < retry_base)
        ):
            raise ValueError("Invalid tracked retry policy")
        source = (
            credential.camera_id,
            credential.worker_generation_id,
            credential.stream_session_id,
        )
        await self.initialize()
        async with self._lock:
            rows = await asyncio.to_thread(
                self._ready_tracked_frames_sync, source, now.timestamp(), limit
            )
        resolved = 0
        for row in rows:
            now = _require_aware_utc(self._clock())
            credential.ensure_current(now)
            if now.timestamp() >= row["expires_at"]:
                continue
            references = TrackedFrameReferences.model_validate_json(row["references_json"])
            try:
                result = await client.read_tracked_frame_result(
                    credential,
                    row["frame_id"],
                    scope_sha256=references.scope_sha256,
                    sequence_number=references.sequence_number,
                    subjects=references.subject_references(),
                    retired_segment_ids=set(references.retired_segment_ids),
                )
            except (httpx.HTTPStatusError, httpx.TransportError, TimeoutError) as error:
                if retry_base is None or retry_max is None or not _tracked_transient(error):
                    raise
                await self.defer_tracked_frame(
                    credential, row["frame_id"], retry_base=retry_base, retry_max=retry_max
                )
                continue
            async with self._lock:
                resolved += await asyncio.to_thread(
                    self._finish_tracked_lookup_sync, source, row, result
                )
        return resolved

    def _ready_tracked_frames_sync(
        self, source: tuple[str, str, str], now: float, limit: int
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """SELECT frame_id, references_json, expires_at FROM tracked_frame_pending
                WHERE camera_id = ? AND generation_id = ? AND session_id = ? AND expires_at > ?
                  AND next_attempt_at <= ?
                  AND (delivery_json IS NULL OR raw_stored = 1) AND references_json <> 'null'
                ORDER BY last_checked, rowid LIMIT ?""",
                (*source, now, now, limit),
            ).fetchall()

    def _finish_tracked_lookup_sync(
        self,
        source: tuple[str, str, str],
        row: sqlite3.Row,
        result: TrackedFrameResult | None,
    ) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _require_aware_utc(self._clock()).timestamp()
            key = (*source, row["frame_id"], row["references_json"], row["expires_at"], now)
            where = """WHERE camera_id = ? AND generation_id = ? AND session_id = ?
                AND frame_id = ? AND references_json = ? AND expires_at = ? AND expires_at > ?"""
            if result is None:
                connection.execute(
                    "UPDATE tracked_frame_pending SET last_checked = ?, attempt_count = 0, "
                    "next_attempt_at = 0 " + where,
                    (now, *key),
                )
                return 0
            deleted = connection.execute("DELETE FROM tracked_frame_pending " + where, key)
            if deleted.rowcount != 1:
                return 0
            self._retain_tracked_frame(connection, row["frame_id"], result)
            return 1

    async def refresh_tracked_frame(
        self,
        client: BackendClient,
        credential: WorkerGenerationCredential,
        frame_id: str,
        *,
        scope_sha256: str,
        sequence_number: str,
        subjects: dict[str, tuple[str, int]],
        retired_segment_ids: set[str],
    ) -> bool:
        """Fetch and retain the seal for this exact sent frame, if still current."""
        result = await client.read_tracked_frame_result(
            credential,
            frame_id,
            scope_sha256=scope_sha256,
            sequence_number=sequence_number,
            subjects=subjects,
            retired_segment_ids=retired_segment_ids,
        )
        if result is None:
            return False

        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._retain_tracked_frame_sync, frame_id, result)

    def _retain_tracked_frame_sync(self, frame_id: str, result: TrackedFrameResult) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._retain_tracked_frame(connection, frame_id, result)

    @staticmethod
    def _retain_tracked_frame(
        connection: sqlite3.Connection, frame_id: str, result: TrackedFrameResult
    ) -> bool:
        row = connection.execute(
            "SELECT frame_id, result_json FROM tracked_frame_head WHERE scope_sha256 = ?",
            (result.scope_sha256,),
        ).fetchone()
        if row is not None:
            previous = TrackedFrameResult.model_validate_json(row["result_json"])
            sequence = int(result.sequence_number)
            previous_sequence = int(previous.sequence_number)
            if sequence < previous_sequence:
                return False
            if sequence == previous_sequence:
                if frame_id != row["frame_id"] or result != previous:
                    raise OutboxConflict("Different tracked seal at the same scope and sequence")
                return True

        connection.execute(
            """INSERT INTO tracked_frame_head (scope_sha256, frame_id, result_json)
            VALUES (?, ?, ?)
            ON CONFLICT (scope_sha256) DO UPDATE SET
                frame_id = excluded.frame_id, result_json = excluded.result_json""",
            (result.scope_sha256, frame_id, result.model_dump_json(by_alias=True)),
        )
        return True

    async def read_tracked_predecessor(
        self, scope_sha256: str, expected_sequence_number: str
    ) -> TrackedFrameResult | None:
        """Read only the latest seal and only for the producer's exact prior sequence."""
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(
                self._read_tracked_predecessor_sync, scope_sha256, expected_sequence_number
            )

    def _read_tracked_predecessor_sync(
        self, scope_sha256: str, expected_sequence_number: str
    ) -> TrackedFrameResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM tracked_frame_head WHERE scope_sha256 = ?",
                (scope_sha256,),
            ).fetchone()
        if row is None:
            return None
        result = TrackedFrameResult.model_validate_json(row["result_json"])
        if (
            result.scope_sha256 != scope_sha256
            or result.sequence_number != expected_sequence_number
        ):
            return None
        return result

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
            if event.event_id != row["event_id"]:
                raise RuntimeError("outbox observation identity does not match stored event ID")
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
                    SELECT event_id, payload_json, payload_hash, attempt_count,
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
        tracked_batch_limit: int | None = None,
    ) -> None:
        if retry_base <= timedelta(0):
            raise ValueError("dispatcher retry_base must be positive")
        if retry_max < retry_base:
            raise ValueError("dispatcher retry_max must be at least retry_base")
        if tracked_batch_limit is not None:
            if type(tracked_batch_limit) is not int or tracked_batch_limit < 1:
                raise ValueError("tracked batch limit must be a positive integer")
            if generation_credential is None:
                raise ValueError("scheduled tracked delivery requires the original credential")
        self._tracked_batch_limit = tracked_batch_limit
        self._outbox = outbox
        self._backend = backend
        self._retry_base = retry_base
        self._retry_max = retry_max
        self._generation_credential = generation_credential

    async def drain_tracked_once(self, *, limit: int) -> int:
        """Send one bounded batch. A failed part leaves the full frame retryable."""
        credential = self._generation_credential
        if credential is None:
            raise OutboxDeliveryBlocked("Tracked delivery requires the original credential")

        stored = 0
        for frame_id, parts in await self._outbox.ready_tracked_parts(credential, limit=limit):
            receipts = []
            try:
                for part in parts:
                    receipts.append(await self._backend.post_tracked_part(credential, part))
            except (httpx.HTTPStatusError, httpx.TransportError, TimeoutError) as error:
                if _tracked_transient(error):
                    await self._outbox.defer_tracked_frame(
                        credential,
                        frame_id,
                        retry_base=self._retry_base,
                        retry_max=self._retry_max,
                    )
                raise
            if await self._outbox.confirm_tracked_parts_stored(
                credential, frame_id, tuple(receipts)
            ):
                stored += 1
        return stored

    async def _run_tracked_cycle(self, limit: int) -> None:
        credential = self._generation_credential
        if credential is None:
            raise OutboxDeliveryBlocked("Tracked delivery requires the original credential")
        try:
            # Resolve previously stored frames even if a later raw send fails.
            await self._outbox.reconcile_tracked_frames(
                self._backend,
                credential,
                limit=limit,
                retry_base=self._retry_base,
                retry_max=self._retry_max,
            )
            await self.drain_tracked_once(limit=limit)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in (408, 429) and not 500 <= status < 600:
                raise
        except (httpx.TransportError, TimeoutError):
            # The next supervised tick retries the original durable bytes.
            return

    async def drain_once(self, *, limit: int = 32) -> int:
        if self._tracked_batch_limit is not None:
            await self._run_tracked_cycle(min(limit, self._tracked_batch_limit))

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
