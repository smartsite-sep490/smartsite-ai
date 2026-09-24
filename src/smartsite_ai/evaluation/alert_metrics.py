"""Deterministic matching and metrics for missing-PPE candidate episodes."""

import math
from collections.abc import Sequence
from datetime import timedelta
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from smartsite_ai.evaluation.detection_metrics import ImmutableMetricsModel, MetricValue
from smartsite_ai.evaluation.models import (
    MAX_IDENTIFIER_LENGTH,
    MAX_SAFE_INTEGER,
    MAX_VIDEO_TIME_SECONDS,
    GroundTruthPpeEpisode,
    PersonInstanceId,
)


class PredictedPpeEpisode(ImmutableMetricsModel):
    """One technical candidate episode associated with a labelled person."""

    candidate_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    clip_id: str = Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH)
    person_instance_id: PersonInstanceId
    stream_session_id: UUID
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"]
    start_time_seconds: float = Field(ge=0.0, le=MAX_VIDEO_TIME_SECONDS)
    end_time_seconds: float = Field(ge=0.0, le=MAX_VIDEO_TIME_SECONDS)
    confirmed_time_seconds: float = Field(ge=0.0, le=MAX_VIDEO_TIME_SECONDS)
    track_id: int = Field(ge=0, le=MAX_SAFE_INTEGER)

    @field_validator("candidate_id", "clip_id")
    @classmethod
    def reject_blank_or_padded_ids(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("candidate_id and clip_id must be non-blank and unpadded")
        return value

    @model_validator(mode="after")
    def validate_episode(self) -> Self:
        if self.person_instance_id is None:
            raise ValueError("person_instance_id is required")
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than or equal to start_time_seconds")
        if not self.start_time_seconds <= self.confirmed_time_seconds <= self.end_time_seconds:
            raise ValueError("confirmed_time_seconds must be within the predicted episode interval")
        return self


class CandidateEpisodeMatch(ImmutableMetricsModel):
    candidate_id: str
    clip_id: str
    person_instance_id: int | str
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"]
    ground_truth_start_seconds: float
    ground_truth_end_seconds: float
    overlap_seconds: float = Field(ge=0.0)
    onset_difference_seconds: float = Field(ge=0.0)


class FragmentationDiagnostic(ImmutableMetricsModel):
    clip_id: str
    person_instance_id: int | str
    ppe_item: Literal["HARD_HAT", "SAFETY_VEST"]
    ground_truth_start_seconds: float
    ground_truth_end_seconds: float
    track_identities: tuple[tuple[str, int], ...]
    additional_track_count: int = Field(ge=1)


class EpisodeMetricsReport(ImmutableMetricsModel):
    onset_tolerance_seconds: float = Field(ge=0.0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    duplicate_count: int = Field(ge=0)
    duplicate_candidate_ids: tuple[str, ...]
    fragmentation_count: int = Field(ge=0)
    fragmentation_diagnostics: tuple[FragmentationDiagnostic, ...]
    raw_false_candidate_count: int = Field(ge=0)
    evaluated_seconds: float = Field(ge=0.0)
    false_candidates_per_camera_hour: MetricValue
    rate_gate_eligible: bool
    rate_gate_ineligibility_reason: str | None
    matches: tuple[CandidateEpisodeMatch, ...]
    false_candidates: tuple[PredictedPpeEpisode, ...]
    missed_episodes: tuple[GroundTruthPpeEpisode, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.true_positives != len(self.matches):
            raise ValueError("true_positives must equal the number of matches")
        if self.false_positives != len(self.false_candidates):
            raise ValueError("false_positives must equal the number of false candidates")
        if self.false_negatives != len(self.missed_episodes):
            raise ValueError("false_negatives must equal the number of missed episodes")
        if self.duplicate_count != len(self.duplicate_candidate_ids):
            raise ValueError("duplicate_count must equal the number of duplicate candidate IDs")
        if self.fragmentation_count != sum(
            item.additional_track_count for item in self.fragmentation_diagnostics
        ):
            raise ValueError("fragmentation_count must equal the diagnostic total")
        if self.raw_false_candidate_count != self.false_positives:
            raise ValueError("raw_false_candidate_count must equal false_positives")
        expected_precision, expected_recall, expected_f1 = _metric_values(
            self.true_positives,
            self.false_positives,
            self.false_negatives,
        )
        if self.precision != expected_precision:
            raise ValueError("precision does not match raw counts")
        if self.recall != expected_recall:
            raise ValueError("recall does not match raw counts")
        if self.f1 != expected_f1:
            raise ValueError("f1 does not match raw counts")
        expected_rate = (
            _defined(self.false_positives * 3600.0 / self.evaluated_seconds)
            if self.evaluated_seconds > 0.0
            else _undefined("evaluated duration is zero")
        )
        if self.false_candidates_per_camera_hour != expected_rate:
            raise ValueError(
                "false_candidates_per_camera_hour does not match raw false candidates and duration"
            )
        if self.rate_gate_eligible != (self.evaluated_seconds >= 1800.0):
            raise ValueError("rate_gate_eligible disagrees with evaluated_seconds")
        if self.rate_gate_eligible == (self.rate_gate_ineligibility_reason is not None):
            raise ValueError(
                "rate_gate_ineligibility_reason must be present exactly when the gate is ineligible"
            )
        return self


def _defined(value: float) -> MetricValue:
    return MetricValue(value=value, reason=None)


def _undefined(reason: str) -> MetricValue:
    return MetricValue(value=None, reason=reason)


def _metric_values(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[MetricValue, MetricValue, MetricValue]:
    predicted = true_positives + false_positives
    ground_truth = true_positives + false_negatives
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    precision = (
        _defined(true_positives / predicted)
        if predicted
        else _undefined("no predicted candidate episodes")
    )
    recall = (
        _defined(true_positives / ground_truth)
        if ground_truth
        else _undefined("no ground-truth candidate episodes")
    )
    f1 = (
        _defined(2 * true_positives / f1_denominator)
        if f1_denominator
        else _undefined("no ground truth or predictions")
    )
    return precision, recall, f1


def _person_sort_key(value: int | str) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _ground_truth_key(
    episode: GroundTruthPpeEpisode,
) -> tuple[str, tuple[str, str], str, float, float]:
    return (
        episode.clip_id,
        _person_sort_key(episode.person_instance_id),
        episode.ppe_item,
        episode.start_time_seconds,
        episode.end_time_seconds,
    )


def _compatible(
    ground_truth: GroundTruthPpeEpisode,
    prediction: PredictedPpeEpisode,
    tolerance_seconds: float,
) -> tuple[bool, float, float]:
    if (
        ground_truth.clip_id != prediction.clip_id
        or ground_truth.person_instance_id != prediction.person_instance_id
        or ground_truth.ppe_item != prediction.ppe_item
    ):
        return False, 0.0, 0.0
    overlap = max(
        0.0,
        min(ground_truth.end_time_seconds, prediction.end_time_seconds)
        - max(ground_truth.start_time_seconds, prediction.start_time_seconds),
    )
    intervals_touch = max(ground_truth.start_time_seconds, prediction.start_time_seconds) <= min(
        ground_truth.end_time_seconds, prediction.end_time_seconds
    )
    onset_difference = abs(ground_truth.start_time_seconds - prediction.start_time_seconds)
    return intervals_touch or onset_difference <= tolerance_seconds, overlap, onset_difference


def match_candidate_episodes(
    ground_truth: Sequence[GroundTruthPpeEpisode],
    predictions: Sequence[PredictedPpeEpisode],
    *,
    onset_tolerance: timedelta,
    evaluated_seconds: float = 0.0,
) -> EpisodeMetricsReport:
    """Match candidate episodes and compute diagnostic false-candidate rates."""

    tolerance_seconds = onset_tolerance.total_seconds()
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise ValueError("onset_tolerance must be finite and non-negative")
    if not math.isfinite(evaluated_seconds) or evaluated_seconds < 0.0:
        raise ValueError("evaluated_seconds must be finite and non-negative")

    candidate_ids = [item.candidate_id for item in predictions]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate_id in predictions")
    ground_truth_keys = [_ground_truth_key(item) for item in ground_truth]
    if len(ground_truth_keys) != len(set(ground_truth_keys)):
        raise ValueError("duplicate ground-truth episode")

    candidates: list[
        tuple[
            float,
            float,
            str,
            tuple[str, tuple[str, str], str, float, float],
            GroundTruthPpeEpisode,
            PredictedPpeEpisode,
        ]
    ] = []
    compatible_by_ground_truth: dict[
        tuple[str, tuple[str, str], str, float, float], list[PredictedPpeEpisode]
    ] = {key: [] for key in ground_truth_keys}
    for episode, episode_key in zip(ground_truth, ground_truth_keys, strict=True):
        for prediction in predictions:
            compatible, overlap, onset_difference = _compatible(
                episode, prediction, tolerance_seconds
            )
            if not compatible:
                continue
            compatible_by_ground_truth[episode_key].append(prediction)
            candidates.append(
                (
                    -overlap,
                    onset_difference,
                    prediction.candidate_id,
                    episode_key,
                    episode,
                    prediction,
                )
            )

    candidates.sort(key=lambda candidate: candidate[:4])
    matched_ground_truth: set[tuple[str, tuple[str, str], str, float, float]] = set()
    matched_predictions: set[str] = set()
    matches: list[CandidateEpisodeMatch] = []
    for negative_overlap, onset_difference, _, episode_key, episode, prediction in candidates:
        if episode_key in matched_ground_truth or prediction.candidate_id in matched_predictions:
            continue
        matched_ground_truth.add(episode_key)
        matched_predictions.add(prediction.candidate_id)
        matches.append(
            CandidateEpisodeMatch(
                candidate_id=prediction.candidate_id,
                clip_id=episode.clip_id,
                person_instance_id=episode.person_instance_id,
                ppe_item=episode.ppe_item,
                ground_truth_start_seconds=episode.start_time_seconds,
                ground_truth_end_seconds=episode.end_time_seconds,
                overlap_seconds=-negative_overlap,
                onset_difference_seconds=onset_difference,
            )
        )

    false_candidates = tuple(
        sorted(
            (item for item in predictions if item.candidate_id not in matched_predictions),
            key=lambda item: item.candidate_id,
        )
    )
    missed_episodes = tuple(
        sorted(
            (item for item in ground_truth if _ground_truth_key(item) not in matched_ground_truth),
            key=_ground_truth_key,
        )
    )
    duplicate_candidate_ids = tuple(
        prediction.candidate_id
        for prediction in false_candidates
        if any(prediction in compatible_by_ground_truth[key] for key in matched_ground_truth)
    )
    fragmentation_diagnostics: list[FragmentationDiagnostic] = []
    ground_truth_by_key = {_ground_truth_key(episode): episode for episode in ground_truth}
    for episode_key in sorted(compatible_by_ground_truth):
        track_identities = tuple(
            sorted(
                {
                    (str(item.stream_session_id), item.track_id)
                    for item in compatible_by_ground_truth[episode_key]
                }
            )
        )
        if len(track_identities) <= 1:
            continue
        episode = ground_truth_by_key[episode_key]
        fragmentation_diagnostics.append(
            FragmentationDiagnostic(
                clip_id=episode.clip_id,
                person_instance_id=episode.person_instance_id,
                ppe_item=episode.ppe_item,
                ground_truth_start_seconds=episode.start_time_seconds,
                ground_truth_end_seconds=episode.end_time_seconds,
                track_identities=track_identities,
                additional_track_count=len(track_identities) - 1,
            )
        )
    fragmentation_count = sum(item.additional_track_count for item in fragmentation_diagnostics)

    true_positives = len(matches)
    false_positives = len(false_candidates)
    false_negatives = len(missed_episodes)
    precision, recall, f1 = _metric_values(true_positives, false_positives, false_negatives)
    false_rate = (
        _defined(false_positives * 3600.0 / evaluated_seconds)
        if evaluated_seconds > 0.0
        else _undefined("evaluated duration is zero")
    )
    rate_gate_eligible = evaluated_seconds >= 1800.0
    return EpisodeMetricsReport(
        onset_tolerance_seconds=tolerance_seconds,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        duplicate_count=len(duplicate_candidate_ids),
        duplicate_candidate_ids=duplicate_candidate_ids,
        fragmentation_count=fragmentation_count,
        fragmentation_diagnostics=tuple(fragmentation_diagnostics),
        raw_false_candidate_count=false_positives,
        evaluated_seconds=evaluated_seconds,
        false_candidates_per_camera_hour=false_rate,
        rate_gate_eligible=rate_gate_eligible,
        rate_gate_ineligibility_reason=(
            None if rate_gate_eligible else "requires at least 1800 evaluated camera-seconds"
        ),
        matches=tuple(matches),
        false_candidates=false_candidates,
        missed_episodes=missed_episodes,
    )


__all__ = [
    "CandidateEpisodeMatch",
    "EpisodeMetricsReport",
    "FragmentationDiagnostic",
    "PredictedPpeEpisode",
    "match_candidate_episodes",
]
