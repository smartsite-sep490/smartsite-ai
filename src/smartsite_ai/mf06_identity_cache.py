"""Short-lived MF06 response metadata only; no images or enrollment templates."""

from collections import OrderedDict
from time import monotonic
from typing import Any

MAX_CACHE_ENTRIES = 32
CACHE_TTL_SECONDS = 60.0


class Mf06IdentityCache:
    def __init__(self, provider: object) -> None:
        self.provider: object | None = provider
        self._closed = False
        self._entries: OrderedDict[str, tuple[str, float, dict[str, Any]]] = OrderedDict()

    def _expire(self) -> float:
        now = monotonic()
        for key, (_, deadline, _) in tuple(self._entries.items()):
            if deadline <= now:
                del self._entries[key]
        return now

    def lookup(self, invocation_id: str, fingerprint: str) -> dict[str, Any] | None:
        self._expire()
        entry = self._entries.get(invocation_id)
        if entry is None:
            return None
        if entry[0] != fingerprint:
            raise ValueError("MF06_INVOCATION_CONFLICT")
        self._entries.move_to_end(invocation_id)
        return dict(entry[2])

    def remember(self, invocation_id: str, fingerprint: str, response: dict[str, Any]) -> None:
        if self._closed or not (
            response.get("state") == "CANDIDATE"
            or (response.get("state") == "UNAVAILABLE" and response.get("reason") == "NO_MATCH")
        ):
            return
        now = self._expire()
        existing = self._entries.get(invocation_id)
        if existing is not None and existing[0] != fingerprint:
            raise ValueError("MF06_INVOCATION_CONFLICT")
        self._entries[invocation_id] = (fingerprint, now + CACHE_TTL_SECONDS, dict(response))
        self._entries.move_to_end(invocation_id)
        while len(self._entries) > MAX_CACHE_ENTRIES:
            self._entries.popitem(last=False)

    def close(self) -> None:
        self._closed = True
        self.provider = None
        self._entries.clear()
