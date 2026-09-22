"""MF05/MF06 orchestration into the locked technical observation event."""

from smartsite_ai.domain.observations import (
    FrameDimensions,
    PersonObservation,
    TechnicalObservationEvent,
)
from smartsite_ai.domain.regions import CameraRegionConfiguration
from smartsite_ai.inference.models import DetectionBatch
from smartsite_ai.pipelines._geometry import to_observation_bounding_box
from smartsite_ai.pipelines.ppe import PpePipeline
from smartsite_ai.pipelines.zones import RestrictedZonePipeline
from smartsite_ai.tracking.protocol import TrackerProtocol


class Mf05Mf06Pipeline:
    """Run tracking, PPE association and zone-entry observation generation."""

    def __init__(
        self,
        *,
        tracker: TrackerProtocol,
        ppe: PpePipeline,
        zones: RestrictedZonePipeline,
        ppe_region_id: str,
    ) -> None:
        if not ppe_region_id:
            raise ValueError("ppe_region_id must not be empty")
        self._tracker = tracker
        self._ppe = ppe
        self._zones = zones
        self._ppe_region_id = ppe_region_id.casefold()

    def process(
        self,
        batch: DetectionBatch,
        *,
        region_configuration: CameraRegionConfiguration,
        event_id: str,
    ) -> TechnicalObservationEvent | None:
        """Create one technical event, or ``None`` when no person is observable."""

        if batch.camera_external_id != region_configuration.camera_external_id:
            raise ValueError(
                "camera region configuration does not match the detection batch camera"
            )
        ppe_region = next(
            (
                region
                for region in region_configuration.regions
                if region.region_id.casefold() == self._ppe_region_id
            ),
            None,
        )
        if ppe_region is None:
            raise ValueError(
                f"PPE observation region {self._ppe_region_id!r} is not in the configuration"
            )

        tracked_frame = self._tracker.update(batch)
        person_observations = tuple(
            PersonObservation.model_validate(
                {
                    "type": "PERSON",
                    "trackId": person.track_id,
                    "confidence": person.detection.confidence,
                    "boundingBox": to_observation_bounding_box(person.bounding_box).to_wire_dict(),
                }
            )
            for person in tracked_frame.persons
        )
        ppe_observations = self._ppe.process(tracked_frame, ppe_region)
        zone_observations = self._zones.process(tracked_frame, region_configuration)
        observations = [*person_observations, *ppe_observations, *zone_observations]
        if not observations:
            return None

        return TechnicalObservationEvent.create(
            event_id=event_id,
            camera_external_id=batch.camera_external_id,
            stream_session_id=str(batch.session_id),
            captured_at=batch.captured_at.isoformat(),
            frame_dimensions=FrameDimensions(width=batch.frame_width, height=batch.frame_height),
            observations=observations,
        )


__all__ = ["Mf05Mf06Pipeline"]
