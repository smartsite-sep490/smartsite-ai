"""Deterministic temporal confirmation, clearing, and cooldown for PPE candidates."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from smartsite_ai.domain.observations import PpeObservation, TechnicalObservationEvent
from smartsite_ai.pipelines.ppe import PPE_ITEMS, PpeItem


@dataclass(frozen=True, slots=True)
class ConfirmedPpeCandidate:
    """An immutable candidate episode confirmed by consecutive missing evidence."""

    stream_id: str
    session_id: UUID
    track_id: int
    ppe_item: PpeItem
    first_seen_at: datetime
    confirmed_at: datetime


@dataclass(slots=True)
class _TrackItemState:
    last_seen_at: datetime
    consecutive_missing: int = 0
    consecutive_clear: int = 0
    first_seen_at: datetime | None = None
    confirmed: bool = False


class TemporalPpeCandidateGate:
    """Evaluate consecutive missing observations against clearing and cooldown policies."""

    def __init__(
        self,
        *,
        confirmation_frames: int = 3,
        clear_frames: int = 2,
        cooldown: timedelta = timedelta(seconds=10),
        track_expiry: timedelta = timedelta(seconds=1),
    ) -> None:
        if confirmation_frames <= 0:
            raise ValueError("confirmation_frames must be positive")
        if clear_frames <= 0:
            raise ValueError("clear_frames must be positive")
        if cooldown < timedelta(0):
            raise ValueError("cooldown must not be negative")
        if track_expiry < timedelta(0):
            raise ValueError("track_expiry must not be negative")

        self._confirmation_frames = confirmation_frames
        self._clear_frames = clear_frames
        self._cooldown = cooldown
        self._track_expiry = track_expiry

        self._last_observed_at: dict[tuple[str, UUID], datetime] = {}
        self._states: dict[tuple[str, UUID, int, PpeItem], _TrackItemState] = {}
        self._cooldowns: dict[tuple[str, UUID, int, PpeItem], datetime] = {}

    def update(
        self,
        *,
        stream_id: str,
        session_id: UUID,
        observed_at: datetime,
        active_track_ids: Sequence[int],
        observations: Sequence[PpeObservation],
    ) -> tuple[ConfirmedPpeCandidate, ...]:
        """Update temporal state and return new confirmed candidates outside cooldown."""

        stream_key = (stream_id, session_id)
        last_observed = self._last_observed_at.get(stream_key)
        if last_observed is not None and observed_at <= last_observed:
            raise ValueError(
                f"observed_at must be strictly increasing for {stream_id}/{session_id}: "
                f"{observed_at} <= {last_observed}"
            )
        self._last_observed_at[stream_key] = observed_at

        # Deduplicate active tracks preserving order
        unique_active_track_ids = tuple(dict.fromkeys(active_track_ids))
        active_track_set = set(unique_active_track_ids)
        for key, state in list(self._states.items()):
            if key[0] == stream_id and key[1] == session_id:
                track_id = key[2]
                if track_id not in active_track_set and (
                    observed_at - state.last_seen_at > self._track_expiry
                ):
                    del self._states[key]

        obs_by_track_item: dict[tuple[int, PpeItem], PpeObservation] = {
            (obs.track_id, obs.ppe_item): obs for obs in observations if obs.type == "PPE"
        }

        confirmed_candidates: list[ConfirmedPpeCandidate] = []
        for track_id in unique_active_track_ids:
            for item in PPE_ITEMS:
                state_key = (stream_id, session_id, track_id, item)
                obs = obs_by_track_item.get((track_id, item))
                state = self._states.get(state_key)

                # Reappearance expiry: if active track reappears after > track_expiry
                # without intermediate frames
                if state is not None and (observed_at - state.last_seen_at > self._track_expiry):
                    state = None
                    self._states.pop(state_key, None)

                if state is None:
                    state = _TrackItemState(last_seen_at=observed_at)
                    self._states[state_key] = state
                state.last_seen_at = observed_at

                if obs is not None and obs.status == "MISSING":
                    state.consecutive_missing += 1
                    state.consecutive_clear = 0
                    if state.first_seen_at is None:
                        state.first_seen_at = observed_at

                    if (
                        state.consecutive_missing >= self._confirmation_frames
                        and not state.confirmed
                    ):
                        last_confirmed = self._cooldowns.get(state_key)
                        if last_confirmed is None or (
                            observed_at - last_confirmed >= self._cooldown
                        ):
                            candidate = ConfirmedPpeCandidate(
                                stream_id=stream_id,
                                session_id=session_id,
                                track_id=track_id,
                                ppe_item=item,
                                first_seen_at=state.first_seen_at,
                                confirmed_at=observed_at,
                            )
                            confirmed_candidates.append(candidate)
                            self._cooldowns[state_key] = observed_at
                            state.confirmed = True
                else:
                    # Non-missing frame (explicit PRESENT or absent/omitted item)
                    state.consecutive_clear += 1
                    if not state.confirmed:
                        # Before confirmation, any non-missing frame immediately resets
                        # the pending missing streak.
                        state.consecutive_missing = 0
                        state.first_seen_at = None
                    elif state.consecutive_clear >= self._clear_frames:
                        # After confirmation, clear_frames consecutive non-missing frames
                        # are required to reset the confirmed episode.
                        state.consecutive_missing = 0
                        state.first_seen_at = None
                        state.confirmed = False

        return tuple(confirmed_candidates)


def filter_event_for_delivery(
    event: TechnicalObservationEvent,
    confirmed_ppe_candidates: Sequence[ConfirmedPpeCandidate]
    | Iterable[ConfirmedPpeCandidate] = (),
) -> TechnicalObservationEvent | None:
    """Filter an event for Backend delivery and timeline recording.

    Returns an event containing:
    - PERSON, ZONE_ENTRY, and other non-PPE observations unchanged.
    - PPE observations with status PRESENT unchanged.
    - PPE observations with status MISSING only when (track_id, ppe_item) matches
      a confirmed candidate belonging to this event's stream session.
    - Returns None if no deliverable trigger (ZONE_ENTRY or confirmed PPE/MISSING) remains.
    """

    event_session_str = event.stream_session_id
    confirmed_keys = {
        (candidate.track_id, candidate.ppe_item)
        for candidate in confirmed_ppe_candidates
        if str(candidate.session_id) == event_session_str
    }

    filtered_observations = []
    has_deliverable_trigger = False

    for obs in event.observations:
        if obs.type == "ZONE_ENTRY":
            filtered_observations.append(obs)
            has_deliverable_trigger = True
        elif obs.type == "PPE":
            if obs.status == "PRESENT":
                filtered_observations.append(obs)
            elif obs.status == "MISSING" and (obs.track_id, obs.ppe_item) in confirmed_keys:
                filtered_observations.append(obs)
                has_deliverable_trigger = True
        else:
            filtered_observations.append(obs)

    if not has_deliverable_trigger or not filtered_observations:
        return None

    return TechnicalObservationEvent.create(
        event_id=event.event_id,
        camera_external_id=event.camera_external_id,
        stream_session_id=event.stream_session_id,
        captured_at=event.captured_at,
        frame_dimensions=event.frame_dimensions,
        observations=filtered_observations,
        evidence=event.evidence,
        schema_version=event.schema_version,
    )


def should_post_event(
    event: TechnicalObservationEvent,
    confirmed_ppe_candidates: Sequence[ConfirmedPpeCandidate]
    | Iterable[ConfirmedPpeCandidate] = (),
) -> bool:
    """Return whether a technical observation event should be posted to the Backend."""

    return filter_event_for_delivery(event, confirmed_ppe_candidates) is not None


__all__ = [
    "ConfirmedPpeCandidate",
    "TemporalPpeCandidateGate",
    "filter_event_for_delivery",
    "should_post_event",
]
