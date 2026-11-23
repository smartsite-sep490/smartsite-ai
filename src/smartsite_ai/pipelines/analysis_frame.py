"""Immutable exact-frame technical preview input; no transport or business verdict."""

from dataclasses import dataclass
from typing import Literal

from smartsite_ai.core.region_configuration_store import CameraBundle
from smartsite_ai.domain.observations import TechnicalObservationEvent
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.ingestion.envelope import FrameEnvelope
from smartsite_ai.pipelines.observation_frame import ObservationFrame


@dataclass(frozen=True, slots=True)
class AnalysisFrame:
    frame: FrameEnvelope
    observation: TechnicalObservationEvent | ObservationFrame | None
    configuration: CameraRegionConfiguration | CameraBundle
    ppe_region_id: str
    confirmed_ppe_items: frozenset[tuple[int, str]]
    occupied_zone_regions: frozenset[tuple[int, str]] = frozenset()
    zone_anchor: Literal["BOTTOM_CENTER", "BOX_CENTER"] = "BOTTOM_CENTER"
