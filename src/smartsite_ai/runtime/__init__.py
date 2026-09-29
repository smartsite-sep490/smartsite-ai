"""Supervised MF05/MF06 runtime for one verified model and 1–3 cameras."""

from smartsite_ai.runtime.inference_lane import SerializingDetector, share_detector
from smartsite_ai.runtime.manifest import (
    CameraRuntimeError,
    bind_manifest,
    load_manifest_bytes,
    parse_manifest,
)
from smartsite_ai.runtime.supervisor import (
    CameraOutcome,
    CameraSession,
    RuntimePlan,
    RuntimeReport,
    execute_runtime,
    exit_code,
    report_payload,
    supervise_sessions,
)

__all__ = [
    "CameraOutcome",
    "CameraRuntimeError",
    "CameraSession",
    "RuntimePlan",
    "RuntimeReport",
    "SerializingDetector",
    "bind_manifest",
    "execute_runtime",
    "exit_code",
    "load_manifest_bytes",
    "parse_manifest",
    "report_payload",
    "share_detector",
    "supervise_sessions",
]
