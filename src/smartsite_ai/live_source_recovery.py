"""Opt-in source-only recovery inside one existing producer/model lifetime."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from smartsite_ai.ingestion.backoff import ExponentialBackoff
from smartsite_ai.ingestion.source import SourceConnectionError


class ReconnectableSource(Protocol):
    async def connect(self) -> None: ...


class LiveSourceRecovery:
    def __init__(
        self,
        attempts: int,
        *,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.maximum = attempts
        self.attempts = 0
        self.fresh_frames = 0
        self.sleeper = sleeper
        self.backoff = ExponentialBackoff(max_delay=4.0)

    def observed_frame(self) -> None:
        self.fresh_frames += 1
        # Only actual fresh reads replenish the outage budget, never viewer polls.
        if self.fresh_frames >= 30:
            self.attempts = 0

    async def reconnect(self, source: ReconnectableSource) -> bool:
        self.fresh_frames = 0
        while self.attempts < self.maximum:
            self.attempts += 1
            await self.sleeper(self.backoff.next_delay(self.attempts - 1))
            try:
                # OpenCvFrameSource.connect closes the old capture and assigns
                # a NEW session/sequence origin. Model/provider are not reloaded.
                await source.connect()
                return True
            except SourceConnectionError:
                continue
        return False
