"""Optional synchronous display tap; observer failure cannot own source cleanup."""

from typing import Protocol

from smartsite_ai.ingestion.envelope import FrameEnvelope


class FrameObserver(Protocol):
    def offer(self, frame: FrameEnvelope) -> None: ...

    def clear(self) -> None: ...


class OptionalFrameObserver:
    def __init__(self, observer: FrameObserver | None) -> None:
        self._observer = observer
        self._failed = False

    @property
    def available(self) -> bool:
        return self._observer is not None and not self._failed

    def offer(self, frame: FrameEnvelope) -> None:
        if self._observer is not None and not self._failed:
            try:
                self._observer.offer(frame)
            except Exception:
                # Do not retain error text: an observer may expose credentials.
                self._failed = True
                self.clear()

    def clear(self) -> None:
        if self._observer is not None:
            try:
                self._observer.clear()
            except Exception:
                self._failed = True
