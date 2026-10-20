from pathlib import Path

from smartsite_ai.domain.regions import (
    MAX_CAMERA_REGION_PAYLOAD_BYTES,
    CameraRegionConfiguration,
)

MAX_RETAINED_REGION_HISTORY = 1024


def read_configuration(path: Path) -> CameraRegionConfiguration:
    """Read a bounded operator snapshot, then apply the existing strict wire parser."""
    with path.open("rb") as source:
        payload = source.read(MAX_CAMERA_REGION_PAYLOAD_BYTES + 1)
    if len(payload) > MAX_CAMERA_REGION_PAYLOAD_BYTES:
        raise ValueError("Region configuration exceeds the wire size limit")
    return CameraRegionConfiguration.from_wire_bytes(payload)


class RealtimeConfigurationState:
    """Validate revisions without changing the camera or selected PPE region.

    Retains up to MAX_RETAINED_REGION_HISTORY distinct region identities in memory
    for the active stream session; process or stream restart resets this history.
    """

    def __init__(self, configuration: CameraRegionConfiguration, selected_region_id: str) -> None:
        self.configuration = configuration
        self._selected = selected_region_id.casefold()
        self._selected_region(configuration)
        regions = {region.region_id.casefold(): region for region in configuration.regions}
        if len(regions) > MAX_RETAINED_REGION_HISTORY:
            raise ValueError("Retained region history exceeds the session limit")
        self._regions = regions

    def _selected_region(self, configuration: CameraRegionConfiguration):
        for region in configuration.regions:
            if region.region_id.casefold() == self._selected:
                return region
        raise ValueError("Selected PPE region is absent")

    def apply(self, replacement: CameraRegionConfiguration) -> bool:
        current = self.configuration
        if replacement.camera_external_id != current.camera_external_id:
            raise ValueError("Region configuration belongs to another camera")
        if replacement.configuration_version < current.configuration_version:
            raise ValueError("Region configuration version decreased")
        selected = self._selected_region(replacement)
        if replacement.configuration_version == current.configuration_version:
            if replacement != current:
                raise ValueError("Region configuration content changed without a new version")
            return False
        for region in replacement.regions:
            previous = self._regions.get(region.region_id.casefold())
            if previous is None:
                continue
            if region.geometry_version < previous.geometry_version:
                raise ValueError("Region geometry version decreased")
            if (
                region.geometry_version == previous.geometry_version
                and region.polygon != previous.polygon
            ):
                raise ValueError("Region polygon changed without a new geometry version")
        next_regions = {
            **self._regions,
            **{region.region_id.casefold(): region for region in replacement.regions},
        }
        if len(next_regions) > MAX_RETAINED_REGION_HISTORY:
            raise ValueError("Retained region history exceeds the session limit")
        reset_selected = (
            selected.geometry_version != self._selected_region(current).geometry_version
        )
        self.configuration = replacement
        self._regions = next_regions
        return reset_selected
