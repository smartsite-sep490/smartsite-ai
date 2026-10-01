"""Versioned demo capture heuristics, not calibrated yaw or liveness detection.

Relative pose comes from five detector landmarks. Only safe reason codes leave
this module; pixels, landmarks and numeric measurements are transient.
"""

import math
from typing import Any, Literal

CaptureTarget = Literal["front", "left", "right"]
QUALITY_ACCEPTED = "FACE_QUALITY_ACCEPTED"


def assess_enrollment_jpeg(analysis: Any, jpeg: bytes, target: CaptureTarget | None) -> str:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return "FACE_IMAGE_INVALID"
    height, width = image.shape[:2]
    faces = analysis.get(image)
    if not faces:
        return "FACE_NOT_FOUND"
    if len(faces) != 1:
        return "FACE_MULTIPLE_FOUND"
    face = faces[0]
    box = np.asarray(face.bbox, dtype=float)
    points = np.asarray(face.kps, dtype=float)
    if (
        box.shape != (4,)
        or points.shape != (5, 2)
        or not np.isfinite(box).all()
        or not np.isfinite(points).all()
    ):
        return "FACE_LANDMARKS_UNAVAILABLE"
    if not math.isfinite(float(face.det_score)) or float(face.det_score) < 0.70:
        return "FACE_NOT_CLEAR"
    x1, y1, x2, y2 = box
    if x1 < 1 or y1 < 1 or x2 >= width - 1 or y2 >= height - 1:
        return "FACE_CLIPPED"
    if x2 - x1 < 160 or y2 - y1 < 160 or (x2 - x1) / width < 0.18:
        return "FACE_TOO_SMALL"
    if (x2 - x1) / width > 0.75 or (y2 - y1) / height > 0.90:
        return "FACE_TOO_CLOSE"
    if abs((x1 + x2) / 2 / width - 0.5) > 0.22 or abs((y1 + y2) / 2 / height - 0.5) > 0.25:
        return "FACE_NOT_CENTERED"
    crop = image[int(y1) : int(y2), int(x1) : int(x2)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 45:
        return "FACE_TOO_DARK"
    if brightness > 215:
        return "FACE_TOO_BRIGHT"
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 45:
        return "FACE_BLURRY"
    if target is None:
        return QUALITY_ACCEPTED
    eyes = sorted(points[:2], key=lambda point: point[0])
    eye_axis = eyes[1] - eyes[0]
    distance = float(np.linalg.norm(eye_axis))
    if distance < 20:
        return "FACE_LANDMARKS_UNAVAILABLE"
    if abs(math.degrees(math.atan2(float(eye_axis[1]), float(eye_axis[0])))) > 18:
        return "FACE_HEAD_TILTED"
    relative_turn = float(
        np.dot(points[2] - (eyes[0] + eyes[1]) / 2, eye_axis) / (distance * distance)
    )
    if abs(relative_turn) > 0.50:
        return "FACE_TURN_TOO_FAR"
    # Raw webcam pixels are unmirrored. User's left is positive in this space;
    # the Web preview is mirrored, so the prompt remains the user's own left.
    correct = (
        abs(relative_turn) <= 0.10
        if target == "front"
        else 0.12 <= relative_turn <= 0.50
        if target == "left"
        else -0.50 <= relative_turn <= -0.12
    )
    return QUALITY_ACCEPTED if correct else f"FACE_POSE_{target.upper()}_REQUIRED"
