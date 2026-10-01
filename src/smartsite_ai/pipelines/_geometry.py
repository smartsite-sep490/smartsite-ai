"""Geometry helpers shared by the PPE and restricted-zone pipelines."""

from smartsite_ai.domain.observations import BoundingBox
from smartsite_ai.inference.models import NormalizedBoundingBox


def to_observation_bounding_box(box: NormalizedBoundingBox) -> BoundingBox:
    """Convert a detector box to the locked technical-observation wire model."""

    return BoundingBox.model_validate(
        {
            "x1": box.x1,
            "y1": box.y1,
            "x2": box.x2,
            "y2": box.y2,
            "coordinateSpace": "NORMALIZED_0_1",
        }
    )


def box_center(box: NormalizedBoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def bottom_center(box: NormalizedBoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, box.y2)


def intersection_over_second(first: NormalizedBoundingBox, second: NormalizedBoundingBox) -> float:
    """Return the fraction of ``second`` covered by ``first``."""

    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / second_area if second_area else 0.0


def point_in_polygon(
    point: tuple[float, float], coordinates: tuple[tuple[float, float], ...]
) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""

    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(coordinates):
        x2, y2 = coordinates[(index + 1) % len(coordinates)]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True

        crosses_horizontal_ray = (y1 > y) != (y2 > y)
        if crosses_horizontal_ray:
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _point_on_segment(
    point: tuple[float, float], first: tuple[float, float], second: tuple[float, float]
) -> bool:
    px, py = point
    x1, y1 = first
    x2, y2 = second
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > 1e-12:
        return False
    return (
        min(x1, x2) - 1e-12 <= px <= max(x1, x2) + 1e-12
        and min(y1, y2) - 1e-12 <= py <= max(y1, y2) + 1e-12
    )


__all__ = [
    "bottom_center",
    "box_center",
    "intersection_over_second",
    "point_in_polygon",
    "to_observation_bounding_box",
]
