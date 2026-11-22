"""Focused RED tests for shared sampled frame continuity policy bounds."""

from __future__ import annotations

from datetime import timedelta

import pytest

from smartsite_ai.tracking.continuity import FrameContinuityPolicy


def test_default_policy_keeps_strict_consecutive_sequence_bounds() -> None:
    policy = FrameContinuityPolicy()

    assert policy.max_source_sequence_gap == 1
    assert policy.allows_sequence(0, 1) is True
    assert policy.allows_sequence(0, 8) is False
    assert policy.allows_sequence(5, 5) is False
    assert policy.allows_sequence(5, 4) is False


def test_sampled_policy_allows_bounded_sequence_gap_and_rejects_overflow_or_rewind() -> None:
    policy = FrameContinuityPolicy(
        max_source_sequence_gap=64,
        max_capture_gap=timedelta(seconds=2),
    )

    assert policy.allows_sequence(0, 50) is True
    assert policy.allows_sequence(50, 56) is True
    assert policy.allows_sequence(0, 64) is True
    assert policy.allows_sequence(0, 65) is False
    assert policy.allows_sequence(50, 50) is False
    assert policy.allows_sequence(50, 49) is False


def test_sampled_policy_rejects_missing_capture_gap_when_sequence_gap_exceeds_one() -> None:
    with pytest.raises(ValueError):
        FrameContinuityPolicy(max_source_sequence_gap=64)

    with pytest.raises(ValueError):
        FrameContinuityPolicy(max_source_sequence_gap=2, max_capture_gap=None)
