"""Bounded exponential backoff with configurable jitter for camera reconnection."""

import random
from collections.abc import Callable


class ExponentialBackoff:
    """Calculates bounded exponential backoff delays with jitter to avoid thundering herd."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        factor: float = 2.0,
        jitter: float = 0.2,
        rng: Callable[[], float] | None = None,
    ) -> None:
        if initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if max_delay < initial_delay:
            raise ValueError("max_delay cannot be smaller than initial_delay")
        if factor < 1.0:
            raise ValueError("factor must be at least 1.0")
        if not (0.0 <= jitter <= 1.0):
            raise ValueError("jitter must be between 0.0 and 1.0")

        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.jitter = jitter
        self._rng = rng if rng is not None else random.random
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def next_delay(self, attempt: int) -> float:
        """Calculate delay for a specific attempt number."""
        nominal = min(self.max_delay, self.initial_delay * (self.factor**attempt))
        if self.jitter > 0.0:
            # Shift uniformly in [-jitter, +jitter]
            uniform_val = self._rng() * 2.0 - 1.0  # in [-1, +1]
            jitter_offset = nominal * self.jitter * uniform_val
            return max(0.0, min(self.max_delay, nominal + jitter_offset))
        return nominal

    def compute_next_delay(self) -> float:
        """Compute the next delay and advance the internal attempt counter."""
        delay = self.next_delay(self._attempts)
        self._attempts += 1
        return delay

    def reset(self) -> None:
        """Reset the internal attempt counter to 0."""
        self._attempts = 0
