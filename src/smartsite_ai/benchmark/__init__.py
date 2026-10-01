"""Local performance-benchmark primitives for the SmartSite detector."""

from smartsite_ai.benchmark.multistream import (
    BenchmarkError,
    DecodedFrame,
    ReplaySourceFacts,
    run_multistream_scenario,
)

__all__ = [
    "BenchmarkError",
    "DecodedFrame",
    "ReplaySourceFacts",
    "run_multistream_scenario",
]
