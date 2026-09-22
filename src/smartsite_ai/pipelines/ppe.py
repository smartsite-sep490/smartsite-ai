"""MF05 PPE-to-person association and technical observation generation."""

from collections.abc import Iterable, Mapping
from typing import Final, Literal

from smartsite_ai.domain.observations import PpeObservation
from smartsite_ai.domain.regions import CameraObservationRegionConfiguration
from smartsite_ai.inference.models import NormalizedBoundingBox, NormalizedDetection
from smartsite_ai.pipelines._geometry import (
    box_center,
    intersection_over_second,
    to_observation_bounding_box,
)
from smartsite_ai.tracking.models import TrackedFrame, TrackedPerson

PpeItem = Literal["HARD_HAT", "SAFETY_VEST"]
PPE_ITEMS: Final[tuple[PpeItem, ...]] = ("HARD_HAT", "SAFETY_VEST")
DEFAULT_PPE_CLASS_NAMES: Final[dict[PpeItem, frozenset[str]]] = {
    "HARD_HAT": frozenset(
        {"hard_hat", "hard hat", "hardhat", "helmet", "safety helmet"}
    ),
    "SAFETY_VEST": frozenset({"safety_vest", "safety vest", "high visibility vest", "vest"}),
}
DEFAULT_MISSING_PPE_CLASS_NAMES: Final[dict[PpeItem, frozenset[str]]] = {
    "HARD_HAT": frozenset({"no-hardhat", "no hardhat", "no_hardhat", "no helmet"}),
    "SAFETY_VEST": frozenset({"no-safety vest", "no safety vest", "no_safety_vest", "no vest"}),
}


class PpePipeline:
    """Associate detector PPE boxes with current person tracks.

    The pipeline emits technical ``PRESENT``/``MISSING`` observations only when the
    detector explicitly identifies an affirmative or negative PPE class. It does not
    decide which PPE a site policy requires or whether a person violated it. An
    undetected item remains unknown by default; callers can opt into legacy
    absence-based observations for a controlled evaluation.
    """

    def __init__(
        self,
        *,
        class_names_by_item: Mapping[PpeItem, Iterable[str]] | None = None,
        missing_class_names_by_item: Mapping[PpeItem, Iterable[str]] | None = None,
        minimum_association_coverage: float = 0.25,
        minimum_observable_person_height: float = 0.1,
        emit_missing: bool = False,
    ) -> None:
        if not 0.0 <= minimum_association_coverage <= 1.0:
            raise ValueError("minimum_association_coverage must be between 0 and 1")
        if not 0.0 < minimum_observable_person_height <= 1.0:
            raise ValueError("minimum_observable_person_height must be greater than 0")

        present_mapping = class_names_by_item or DEFAULT_PPE_CLASS_NAMES
        missing_mapping = missing_class_names_by_item or DEFAULT_MISSING_PPE_CLASS_NAMES
        class_to_observation: dict[str, tuple[PpeItem, Literal["PRESENT", "MISSING"]]] = {}
        for status, source_mapping in (("PRESENT", present_mapping), ("MISSING", missing_mapping)):
            for item, names in source_mapping.items():
                if item not in PPE_ITEMS:
                    raise ValueError(f"unsupported PPE item: {item}")
                for name in names:
                    normalized_name = name.casefold().strip()
                    if not normalized_name:
                        raise ValueError("PPE class names must not be empty")
                    candidate = (item, status)
                    previous = class_to_observation.setdefault(normalized_name, candidate)
                    if previous != candidate:
                        raise ValueError(f"PPE class name maps to multiple observations: {name}")

        self._class_to_observation = class_to_observation
        self._items = tuple(
            item for item in PPE_ITEMS if item in present_mapping or item in missing_mapping
        )
        self._minimum_association_coverage = minimum_association_coverage
        self._minimum_observable_person_height = minimum_observable_person_height
        self._emit_missing = emit_missing

    def process(
        self,
        tracked_frame: TrackedFrame,
        observation_region: CameraObservationRegionConfiguration,
    ) -> tuple[PpeObservation, ...]:
        """Build PPE observations using the supplied immutable region context."""

        associations = self._associate(tracked_frame)
        observations: list[PpeObservation] = []
        for person in tracked_frame.persons:
            for item in self._items:
                association = associations.get((person.track_id, item))
                if association is not None:
                    status, detection = association
                    observations.append(
                        self._observation(
                            person.track_id,
                            item,
                            status,
                            observation_region,
                            detection,
                        )
                    )
                elif self._emit_missing and self._is_observable(person):
                    observations.append(
                        self._observation(
                            person.track_id,
                            item,
                            "MISSING",
                            observation_region,
                            None,
                        )
                    )
        return tuple(observations)

    def _associate(
        self, tracked_frame: TrackedFrame
    ) -> dict[tuple[int, PpeItem], tuple[Literal["PRESENT", "MISSING"], NormalizedDetection]]:
        best: dict[
            tuple[int, PpeItem],
            tuple[float, float, int, Literal["PRESENT", "MISSING"], NormalizedDetection],
        ] = {}
        for detection_index, detection in enumerate(tracked_frame.batch.detections):
            observation = self._class_to_observation.get(detection.class_name.casefold())
            if observation is None:
                continue
            item, status = observation

            candidates: list[tuple[float, float, int, TrackedPerson]] = []
            for person in tracked_frame.persons:
                coverage = intersection_over_second(person.bounding_box, detection.bounding_box)
                detection_center = box_center(detection.bounding_box)
                if coverage < self._minimum_association_coverage:
                    continue
                if not _contains(person.bounding_box, detection_center):
                    continue
                candidates.append((coverage, person.detection.confidence, -person.track_id, person))

            if not candidates:
                continue
            coverage, _person_confidence, _negative_track_id, person = max(candidates)
            key = (person.track_id, item)
            candidate = (coverage, detection.confidence, -detection_index, status, detection)
            current = best.get(key)
            if current is None or candidate[:3] > current[:3]:
                best[key] = candidate

        return {key: (value[3], value[4]) for key, value in best.items()}

    def _is_observable(self, person: TrackedPerson) -> bool:
        box = person.bounding_box
        return (
            box.x1 > 0.0
            and box.y1 > 0.0
            and box.x2 < 1.0
            and box.y2 < 1.0
            and box.y2 - box.y1 >= self._minimum_observable_person_height
        )

    @staticmethod
    def _observation(
        track_id: int,
        item: PpeItem,
        status: Literal["PRESENT", "MISSING"],
        region: CameraObservationRegionConfiguration,
        detection: NormalizedDetection | None,
    ) -> PpeObservation:
        payload: dict[str, object] = {
            "type": "PPE",
            "trackId": track_id,
            "ppeItem": item,
            "status": status,
            "regionId": region.region_id,
            "geometryVersion": region.geometry_version,
        }
        if detection is not None:
            payload["confidence"] = detection.confidence
            payload["boundingBox"] = to_observation_bounding_box(
                detection.bounding_box
            ).to_wire_dict()
        return PpeObservation.model_validate(payload)


def _contains(box: NormalizedBoundingBox, point: tuple[float, float]) -> bool:
    x = point[0]
    y = point[1]
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


__all__ = [
    "DEFAULT_MISSING_PPE_CLASS_NAMES",
    "DEFAULT_PPE_CLASS_NAMES",
    "PPE_ITEMS",
    "PpeItem",
    "PpePipeline",
]
