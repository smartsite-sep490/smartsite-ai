"""Bounded transient face continuity, never identity or access authorization."""

from dataclasses import dataclass, field


@dataclass
class _Entry:
    seen: float
    anchor: list[float] | None = field(default=None, repr=False)
    pending: list[float] | None = field(default=None, repr=False)
    absent_since: float | None = None


class GateContinuity:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def absent(self, session: str, now: float) -> None:
        self.expire(now)
        entry = self._entries.get(session)
        if entry is not None:
            if entry.absent_since is None:
                entry.absent_since = now
            elif now - entry.absent_since >= 2:
                self._entries.pop(session, None)

    def observe(self, session: str, embedding: list[float], now: float) -> str:
        self.expire(now)
        if session not in self._entries:
            if len(self._entries) >= 128:
                self._entries.pop(min(self._entries, key=lambda key: self._entries[key].seen))
            self._entries[session] = _Entry(seen=now)
        entry = self._entries[session]
        entry.seen = now
        entry.absent_since = None

        def cosine(other: list[float]) -> float:
            if len(other) != len(embedding):
                raise ValueError("incompatible continuity dimensions")
            return sum(a * b for a, b in zip(other, embedding, strict=True))

        if entry.anchor is not None:
            score = cosine(entry.anchor)
            if score >= 0.55:
                entry.pending = None
                return "FACE_PRESENCE_SAME"
            if score >= 0.45:
                entry.pending = None
                return "FACE_MATCH_UNCERTAIN"
        if entry.pending is None or cosine(entry.pending) < 0.55:
            entry.pending = embedding
            return "FACE_PRESENCE_STABILIZING"
        entry.anchor = embedding
        entry.pending = None
        return "FACE_PRESENCE_NEW"

    def expire(self, now: float) -> None:
        self._entries = {
            key: value for key, value in self._entries.items() if now - value.seen < 30
        }
