"""Strict evaluation reports, accuracy-gate decisions, and atomic JSON persistence."""

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from smartsite_ai.evaluation.alert_metrics import EpisodeMetricsReport
from smartsite_ai.evaluation.detection_metrics import DetectionMetricsReport
from smartsite_ai.evaluation.models import (
    CANONICAL_PPE_CLASSES,
    CANONICAL_PPE_CLASSES_SET,
    FrozenStringMap,
    ImmutableStringMapping,
)
from smartsite_ai.ingestion.source import sanitize_message_credentials

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_SHA_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
_SENSITIVE_KEY_SUFFIXES = (
    "token",
    "password",
    "credential",
    "authorization",
    "apikey",
    "signedurl",
)


def _sanitize_failure_message(message: str) -> str:
    lines = str(message).splitlines()
    first_line = lines[0] if lines else "operation failed"
    sanitized = sanitize_message_credentials(first_line)
    sanitized = re.sub(r"(?i)\bbearer\s+\S+", "Bearer ***", sanitized)
    sanitized = re.sub(
        r"(?i)\b(token|password|credential|authorization|api[\s_-]?key|signed[\s_-]?url)"
        r"\b.*$",
        lambda match: f"{match.group(1)}=***",
        sanitized,
    )
    return sanitized.strip()[:512] or "operation failed"


def _to_camel(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


class _ReportModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        alias_generator=_to_camel,
    )


def _freeze_string_mapping(value: Mapping[str, str]) -> FrozenStringMap:
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError("mapping keys and values must be strings")
        if not key.strip() or key != key.strip() or not item.strip() or item != item.strip():
            raise ValueError("mapping keys and values must be non-blank and unpadded")
    return FrozenStringMap(value)


def _validate_safe_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain credentials")
    if parse_qsl(parsed.query, keep_blank_values=True):
        raise ValueError("URL must not contain a sensitive query parameter")
    if parsed.fragment:
        raise ValueError("URL must not contain a fragment")
    return value


class GitReportMetadata(_ReportModel):
    commit_sha: str = Field(pattern=_GIT_SHA_PATTERN)
    dirty_worktree: bool


class ModelReportMetadata(_ReportModel):
    artifact_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    family: str = Field(min_length=1, max_length=64)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    source_url: str = Field(min_length=1, max_length=2048)
    license: str = Field(min_length=1, max_length=128)
    class_map: ImmutableStringMapping

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        return _validate_safe_url(value)

    @model_validator(mode="after")
    def validate_class_map(self) -> Self:
        if (
            len(self.class_map) != len(CANONICAL_PPE_CLASSES)
            or set(self.class_map.values()) != CANONICAL_PPE_CLASSES_SET
        ):
            raise ValueError("class_map must contain exactly the five canonical PPE classes")
        object.__setattr__(self, "class_map", _freeze_string_mapping(self.class_map))
        return self


class DatasetReportMetadata(_ReportModel):
    dataset_id: str = Field(min_length=1, max_length=128)
    dataset_version: str = Field(min_length=1, max_length=64)
    aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: Literal["train", "validation", "test"]
    license: str = Field(min_length=1, max_length=128)


class RuntimeReportMetadata(_ReportModel):
    python_version: str = Field(min_length=1, max_length=64)
    platform: str = Field(min_length=1, max_length=256)
    device: str = Field(min_length=1, max_length=128)
    package_versions: ImmutableStringMapping
    hardware: ImmutableStringMapping

    @model_validator(mode="after")
    def freeze_mappings(self) -> Self:
        object.__setattr__(self, "package_versions", _freeze_string_mapping(self.package_versions))
        object.__setattr__(self, "hardware", _freeze_string_mapping(self.hardware))
        return self


class ThresholdReportMetadata(_ReportModel):
    confidence: float = Field(ge=0.0, le=1.0)
    nms_iou: float = Field(gt=0.0, le=1.0)
    match_iou: float = Field(gt=0.0, le=1.0)


class ImageSize(_ReportModel):
    width: int = Field(ge=1, le=16384)
    height: int = Field(ge=1, le=16384)


class CommandArgument(_ReportModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    value: str = Field(max_length=2048)

    @model_validator(mode="after")
    def reject_sensitive_argument(self) -> Self:
        normalized = _normalized_key(self.name)
        if any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES):
            raise ValueError("command argument name must not identify a secret")
        if _sanitize_failure_message(self.value) != self.value:
            raise ValueError("command argument value must not contain credentials")
        return self


class AccuracyGateResult(_ReportModel):
    status: Literal["PASS", "FAIL"]
    reasons: tuple[str, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "PASS" and self.reasons:
            raise ValueError("PASS accuracy gate must not include reasons")
        if self.status == "FAIL" and not self.reasons:
            raise ValueError("FAIL accuracy gate requires at least one reason")
        return self


class FailureDetails(_ReportModel):
    error_type: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.]*$")
    message: str = Field(min_length=1, max_length=512)

    @field_validator("message")
    @classmethod
    def reject_traceback(cls, value: str) -> str:
        if "\n" in value or "\r" in value or "traceback" in value.lower():
            raise ValueError("failure message must be a single safe line without traceback")
        if _sanitize_failure_message(value) != value:
            raise ValueError("failure message contains sensitive data")
        return value


def evaluate_accuracy_gate(metrics: DetectionMetricsReport) -> AccuracyGateResult:
    """Evaluate the locked academic accuracy gate from reported detection metrics."""

    reasons: list[str] = []
    by_class: dict[str, Any] = {}
    for item in metrics.per_class:
        if item.class_name in by_class:
            reasons.append(f"duplicate canonical class: {item.class_name}")
        else:
            by_class[item.class_name] = item

    for class_name in CANONICAL_PPE_CLASSES:
        item = by_class.get(class_name)
        if item is None:
            reasons.append(f"missing canonical class: {class_name}")
        elif item.support < 30:
            reasons.append(f"{class_name} support {item.support} is below 30")

    raw_f1_values = []
    for class_name in CANONICAL_PPE_CLASSES:
        item = by_class.get(class_name)
        if item is None:
            continue
        denominator = 2 * item.true_positives + item.false_positives + item.false_negatives
        if denominator > 0:
            raw_f1_values.append(2 * item.true_positives / denominator)

    macro_f1 = metrics.macro.f1
    raw_macro_f1 = (
        sum(raw_f1_values) / len(raw_f1_values)
        if len(raw_f1_values) == len(CANONICAL_PPE_CLASSES)
        else None
    )
    if macro_f1.value is None:
        reasons.append(f"macro F1 is undefined: {macro_f1.reason}")
    else:
        if raw_macro_f1 is not None and not math.isclose(
            macro_f1.value, raw_macro_f1, rel_tol=0.0, abs_tol=1e-12
        ):
            reasons.append("reported macro F1 does not match raw counts")
        if macro_f1.value < 0.70:
            reasons.append(f"macro F1 {macro_f1.value} is below 0.70")
    if raw_macro_f1 is not None and raw_macro_f1 < 0.70:
        reasons.append(f"macro F1 from raw counts {raw_macro_f1} is below 0.70")

    for class_name in ("NO-Hardhat", "NO-Safety Vest"):
        item = by_class.get(class_name)
        if item is None:
            continue
        if item.recall.value is None:
            reasons.append(f"{class_name} recall is undefined: {item.recall.reason}")
        else:
            raw_recall = item.true_positives / item.support if item.support else None
            if raw_recall is not None and not math.isclose(
                item.recall.value, raw_recall, rel_tol=0.0, abs_tol=1e-12
            ):
                reasons.append(f"reported {class_name} recall does not match raw counts")
            if item.recall.value < 0.75:
                reasons.append(f"{class_name} recall {item.recall.value} is below 0.75")
            if raw_recall is not None and raw_recall < 0.75:
                reasons.append(f"{class_name} recall from raw counts {raw_recall} is below 0.75")

    return AccuracyGateResult(
        status="FAIL" if reasons else "PASS",
        reasons=tuple(reasons),
    )


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _reject_sensitive_keys(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalized_key(str(key))
            if any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES):
                raise ValueError(f"sensitive field name is forbidden at {path}.{key}")
            _reject_sensitive_keys(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_sensitive_keys(item, f"{path}[{index}]")


class EvaluationReport(_ReportModel):
    schema_version: Literal["1.0.0"]
    run_id: UUID
    status: Literal["COMPLETE", "INCOMPLETE"]
    started_at_utc: datetime
    completed_at_utc: datetime
    git: GitReportMetadata
    model: ModelReportMetadata
    dataset: DatasetReportMetadata
    runtime: RuntimeReportMetadata
    thresholds: ThresholdReportMetadata
    image_size: ImageSize
    metric_definitions: tuple[str, ...] = Field(min_length=1, max_length=32)
    command_arguments: tuple[CommandArgument, ...] = Field(max_length=64)
    detection_metrics: DetectionMetricsReport | None
    episode_metrics: EpisodeMetricsReport | None
    accuracy_gate: AccuracyGateResult | None
    failure: FailureDetails | None

    @field_validator("started_at_utc", "completed_at_utc")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("report timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("metric_definitions")
    @classmethod
    def validate_metric_definitions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or value != value.strip() or len(value) > 512 for value in values):
            raise ValueError("metric definitions must be non-blank, unpadded, and bounded")
        return values

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("completed_at_utc must not be before started_at_utc")
        if self.status == "COMPLETE":
            if self.failure is not None:
                raise ValueError("COMPLETE report must not include failure")
            if (
                self.detection_metrics is None
                or self.episode_metrics is None
                or self.accuracy_gate is None
            ):
                raise ValueError("COMPLETE report requires detection, episode, and gate results")
            expected_gate = evaluate_accuracy_gate(self.detection_metrics)
            if self.accuracy_gate != expected_gate:
                raise ValueError("accuracy_gate must be derived from detection_metrics")
        else:
            if self.failure is None:
                raise ValueError("INCOMPLETE report requires failure")
            if self.accuracy_gate is not None:
                raise ValueError("INCOMPLETE report must not claim an accuracy gate result")

        argument_names = [argument.name for argument in self.command_arguments]
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("command argument names must be unique")

        _reject_sensitive_keys(self.model_dump(mode="json", by_alias=True))
        return self


def safe_failure_details(error: BaseException) -> FailureDetails:
    """Convert an exception to bounded, single-line, credential-sanitized details."""

    message = _sanitize_failure_message(str(error))
    return FailureDetails(error_type=type(error).__name__[:128], message=message)


class EvaluationReportWriteError(Exception):
    """Safe report persistence failure."""


def _validate_output_path(path: Path, report: EvaluationReport) -> None:
    required_suffix = ".report.json" if report.status == "COMPLETE" else ".incomplete.json"
    if not path.name.endswith(required_suffix):
        raise EvaluationReportWriteError(
            f"{report.status} report path must end with {required_suffix}"
        )
    if path.exists():
        raise EvaluationReportWriteError("evaluation report output already exists")
    if not path.parent.is_dir():
        raise EvaluationReportWriteError("evaluation report output directory does not exist")


def write_evaluation_report(path: Path, report: EvaluationReport) -> None:
    """Write deterministic JSON through a same-directory temporary file and atomic replace."""

    _validate_output_path(path, report)
    payload = report.model_dump(mode="json", by_alias=True, exclude_none=False)
    try:
        _reject_sensitive_keys(payload)
        serialized = (
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise EvaluationReportWriteError("could not serialize safe evaluation report") from error

    descriptor = -1
    temp_path: Path | None = None
    try:
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as error:
            raise EvaluationReportWriteError("evaluation report output already exists") from error
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        temp_path = None
    except EvaluationReportWriteError:
        raise
    except OSError as error:
        raise EvaluationReportWriteError("could not finalize evaluation report") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


__all__ = [
    "AccuracyGateResult",
    "CommandArgument",
    "DatasetReportMetadata",
    "EvaluationReport",
    "EvaluationReportWriteError",
    "FailureDetails",
    "GitReportMetadata",
    "ImageSize",
    "ModelReportMetadata",
    "RuntimeReportMetadata",
    "ThresholdReportMetadata",
    "evaluate_accuracy_gate",
    "safe_failure_details",
    "write_evaluation_report",
]
