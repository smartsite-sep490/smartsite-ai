"""MF06 restricted-region entry detection without authorization decisions."""

from uuid import UUID

from smartsite_ai.domain.observations import ZoneEntryObservation
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.pipelines._geometry import bottom_center, point_in_polygon
from smartsite_ai.tracking.models import TrackedFrame


class RestrictedZonePipeline:
    """Emit technical entries when a person track transitions into a region.

    The pipeline treats every region in the supplied camera configuration as an
    observation region. It never emits allowed/denied/unauthorized outcomes; those
    decisions remain in the SmartSite backend.
    """

    def __init__(self) -> None:
        self._source_key: tuple[str, UUID] | None = None
        self._last_sequence: int | None = None
        self._inside: dict[tuple[int, str], bool] = {}
        self._region_versions: dict[str, int] = {}

    def process(
        self,
        tracked_frame: TrackedFrame,
        configuration: CameraRegionConfiguration,
    ) -> tuple[ZoneEntryObservation, ...]:
        """Return only outside-to-inside transitions for current tracks."""

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
            self._region_versions.clear()
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

        observations: list[ZoneEntryObservation] = []
        for person in tracked_frame.persons:
            point = bottom_center(person.bounding_box)
            for region_key in sorted(current_regions):
                region = current_regions[region_key]
                inside = point_in_polygon(point, region.polygon.coordinates)
                state_key = (person.track_id, region_key)
                previous = self._inside.get(state_key)
                if previous is False and inside:
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
                self._inside[state_key] = inside

        self._last_sequence = batch.sequence_number
        return tuple(observations)

    def _clear_region_state(self, region_key: str) -> None:
        for state_key in tuple(self._inside):
            if state_key[1] == region_key:
                del self._inside[state_key]


__all__ = ["RestrictedZonePipeline"]
