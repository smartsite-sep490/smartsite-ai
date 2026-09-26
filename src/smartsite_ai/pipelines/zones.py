"""MF06 restricted-region entry detection without authorization decisions."""

from uuid import UUID

from smartsite_ai.domain.observations import ZoneEntryObservation
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.pipelines._geometry import bottom_center, point_in_polygon
from smartsite_ai.tracking.models import TrackedFrame


class RestrictedZonePipeline:
    """Emit confirmed technical entries or initial inside occupancy for a region.

    The pipeline treats every region in the supplied camera configuration as an
    observation region. It never emits allowed/denied/unauthorized outcomes; those
    decisions remain in the SmartSite backend.
    """

    def __init__(
        self,
        *,
        entry_confirmation_frames: int = 3,
        exit_confirmation_frames: int = 2,
    ) -> None:
        if entry_confirmation_frames <= 0:
            raise ValueError("entry_confirmation_frames must be positive")
        if exit_confirmation_frames <= 0:
            raise ValueError("exit_confirmation_frames must be positive")
        self._entry_confirmation_frames = entry_confirmation_frames
        self._exit_confirmation_frames = exit_confirmation_frames
        self._source_key: tuple[str, UUID] | None = None
        self._last_sequence: int | None = None
        self._inside: dict[tuple[int, str], bool] = {}
        self._inside_streaks: dict[tuple[int, str], int] = {}
        self._outside_streaks: dict[tuple[int, str], int] = {}
        self._absent_frames: dict[int, int] = {}
        self._region_versions: dict[str, int] = {}
        self._region_ids: dict[str, str] = {}

    def process(
        self,
        tracked_frame: TrackedFrame,
        configuration: CameraRegionConfiguration,
    ) -> tuple[ZoneEntryObservation, ...]:
        """Return entries after consecutive inside evidence for current tracks.

        Initial inside occupancy is emitted after the same confirmation window so a
        reconnect or temporary occlusion cannot silently hide an occupied region.
        Consecutive outside evidence rearms a confirmed track and damps boundary jitter.
        """

        batch = tracked_frame.batch
        if batch.camera_external_id != configuration.camera_external_id:
            raise ValueError(
                "camera region configuration does not match the detection batch camera"
            )

        source_key = (batch.stream_id, batch.session_id)
        if self._source_key != source_key:
            self._source_key = source_key
            self._last_sequence = None
            self._inside.clear()
            self._inside_streaks.clear()
            self._outside_streaks.clear()
            self._absent_frames.clear()
            self._region_versions.clear()
            self._region_ids.clear()
        elif self._last_sequence is not None and batch.sequence_number <= self._last_sequence:
            raise ValueError(
                "Tracked frames must be supplied in strictly increasing sequence order"
            )

        current_regions = {region.region_id.casefold(): region for region in configuration.regions}
        for region_key, previous_version in tuple(self._region_versions.items()):
            current_region = current_regions.get(region_key)
            if current_region is None or current_region.geometry_version != previous_version:
                self._clear_region_state(region_key)
        self._region_versions = {
            region_key: region.geometry_version for region_key, region in current_regions.items()
        }
        self._region_ids = {
            region_key: region.region_id for region_key, region in current_regions.items()
        }

        observations: list[ZoneEntryObservation] = []
        for person in tracked_frame.persons:
            point = bottom_center(person.bounding_box)
            for region_key in sorted(current_regions):
                region = current_regions[region_key]
                inside = point_in_polygon(point, region.polygon.coordinates)
                state_key = (person.track_id, region_key)
                previous = self._inside.get(state_key)
                if inside:
                    self._inside_streaks[state_key] = self._inside_streaks.get(state_key, 0) + 1
                    self._outside_streaks[state_key] = 0
                else:
                    self._inside_streaks[state_key] = 0
                    self._outside_streaks[state_key] = self._outside_streaks.get(state_key, 0) + 1

                if (
                    inside
                    and previous is not True
                    and self._inside_streaks[state_key] >= self._entry_confirmation_frames
                ):
                    observations.append(
                        ZoneEntryObservation.model_validate(
                            {
                                "type": "ZONE_ENTRY",
                                "trackId": person.track_id,
                                "regionId": region.region_id,
                                "geometryVersion": region.geometry_version,
                                "confidence": person.detection.confidence,
                            }
                        )
                    )
                    self._inside[state_key] = True
                elif not inside and previous is None:
                    # Initial outside evidence establishes the baseline immediately;
                    # entry still requires consecutive inside evidence.
                    self._inside[state_key] = False
                elif (
                    not inside
                    and previous is True
                    and self._outside_streaks[state_key] >= self._exit_confirmation_frames
                ):
                    self._inside[state_key] = False

        # The tracker can revive an id for two missed frames. Drop zone memory only
        # after that window so a long session cannot keep every historical id.
        active_track_ids = {person.track_id for person in tracked_frame.persons}
        known_state_keys = (
            set(self._inside) | set(self._inside_streaks) | set(self._outside_streaks)
        )
        known_track_ids = {state_key[0] for state_key in known_state_keys}
        for track_id in known_track_ids:
            if track_id in active_track_ids:
                self._absent_frames.pop(track_id, None)
                continue
            for state_key in known_state_keys:
                if state_key[0] == track_id:
                    # Missing frames are not consecutive inside/outside evidence.
                    self._inside_streaks[state_key] = 0
                    self._outside_streaks[state_key] = 0
            missed = self._absent_frames.get(track_id, 0) + 1
            if missed <= 2:
                self._absent_frames[track_id] = missed
                continue
            for state_key in tuple(known_state_keys):
                if state_key[0] == track_id:
                    self._inside.pop(state_key, None)
                    self._inside_streaks.pop(state_key, None)
                    self._outside_streaks.pop(state_key, None)
            self._absent_frames.pop(track_id, None)

        self._last_sequence = batch.sequence_number
        return tuple(observations)

    def occupied_track_regions(
        self,
        *,
        stream_id: str,
        session_id: UUID,
        active_track_ids: tuple[int, ...],
    ) -> frozenset[tuple[int, str]]:
        """Return confirmed current occupancy without emitting another entry event."""

        if self._source_key != (stream_id, session_id):
            return frozenset()
        active = set(active_track_ids)
        return frozenset(
            (track_id, self._region_ids[region_key])
            for (track_id, region_key), inside in self._inside.items()
            if inside and track_id in active and region_key in self._region_ids
        )

    def _clear_region_state(self, region_key: str) -> None:
        state_keys = set(self._inside) | set(self._inside_streaks) | set(self._outside_streaks)
        for state_key in state_keys:
            if state_key[1] == region_key:
                self._inside.pop(state_key, None)
                self._inside_streaks.pop(state_key, None)
                self._outside_streaks.pop(state_key, None)


__all__ = ["RestrictedZonePipeline"]
