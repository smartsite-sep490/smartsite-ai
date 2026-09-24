import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from smartsite_ai.evaluation.alert_metrics import match_candidate_episodes
from smartsite_ai.evaluation.detection_metrics import (
    ClassDetectionMetrics,
    DetectionCounts,
    DetectionMetricsReport,
    MacroDetectionMetrics,
    MetricValue,
)
from smartsite_ai.evaluation.models import CANONICAL_PPE_CLASSES
from smartsite_ai.evaluation.report import (
    AccuracyGateResult,
    CommandArgument,
    DatasetReportMetadata,
    EvaluationReport,
    EvaluationReportWriteError,
    FailureDetails,
    GitReportMetadata,
    ImageSize,
    ModelReportMetadata,
    RuntimeReportMetadata,
    ThresholdReportMetadata,
    evaluate_accuracy_gate,
    safe_failure_details,
    write_evaluation_report,
)

NOW = datetime(2026, 9, 24, 8, 0, tzinfo=UTC)


def _defined(value: float) -> MetricValue:
    return MetricValue(value=value, reason=None)


def _undefined(reason: str) -> MetricValue:
    return MetricValue(value=None, reason=reason)


def _detection_metrics(
    *,
    support: int = 40,
    macro_f1: float | None = 0.70,
    no_hardhat_recall: float | None = 0.75,
    no_safety_vest_recall: float | None = 0.75,
    omit_class: str | None = None,
) -> DetectionMetricsReport:
    per_class = []
    for class_name in CANONICAL_PPE_CLASSES:
        if class_name == omit_class:
            continue
        if support == 40 and class_name in {"NO-Hardhat", "NO-Safety Vest"}:
            true_positives, false_positives, false_negatives = 30, 10, 10
            default_recall, precision_value, f1_value = 0.75, 0.75, 0.75
        elif support == 40:
            true_positives, false_positives, false_negatives = 20, 0, 20
            default_recall, precision_value, f1_value = 0.5, 1.0, 2 / 3
        else:
            true_positives, false_positives, false_negatives = support, 0, 0
            default_recall, precision_value, f1_value = 1.0, 1.0, 1.0
        recall_value = default_recall
        if class_name == "NO-Hardhat":
            recall_value = no_hardhat_recall
        elif class_name == "NO-Safety Vest":
            recall_value = no_safety_vest_recall
        recall = (
            _defined(recall_value)
            if recall_value is not None
            else _undefined("required recall unavailable")
        )
        per_class.append(
            ClassDetectionMetrics(
                class_name=class_name,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                support=support,
                precision=_defined(precision_value),
                recall=recall,
                f1=_defined(f1_value),
            )
        )
    macro_value = _defined(macro_f1) if macro_f1 is not None else _undefined("macro F1 unavailable")
    total_true_positives = sum(item.true_positives for item in per_class)
    total_false_positives = sum(item.false_positives for item in per_class)
    total_false_negatives = sum(item.false_negatives for item in per_class)
    total_support = sum(item.support for item in per_class)
    micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    micro_recall = total_true_positives / total_support
    micro_f1 = (
        2
        * total_true_positives
        / (2 * total_true_positives + total_false_positives + total_false_negatives)
    )
    macro_precision = sum(item.precision.value for item in per_class) / len(per_class)
    defined_recalls = [item.recall.value for item in per_class if item.recall.value is not None]
    macro_recall = (
        _defined(sum(defined_recalls) / len(defined_recalls))
        if len(defined_recalls) == len(per_class)
        else _undefined("one or more class recalls unavailable")
    )
    return DetectionMetricsReport(
        iou_threshold=0.5,
        per_class=tuple(per_class),
        macro=MacroDetectionMetrics(
            supported_classes=tuple(item.class_name for item in per_class),
            excluded_classes=(),
            precision=_defined(macro_precision),
            recall=macro_recall,
            f1=macro_value,
        ),
        micro=DetectionCounts(
            true_positives=total_true_positives,
            false_positives=total_false_positives,
            false_negatives=total_false_negatives,
            support=total_support,
            precision=_defined(micro_precision),
            recall=_defined(micro_recall),
            f1=_defined(micro_f1),
        ),
    )


def _complete_report(
    *,
    detection_metrics: DetectionMetricsReport | None = None,
    hardware: dict[str, str] | None = None,
    package_versions: dict[str, str] | None = None,
) -> EvaluationReport:
    metrics = detection_metrics or _detection_metrics()
    return EvaluationReport(
        schema_version="1.0.0",
        run_id=UUID("10000000-0000-0000-0000-000000000001"),
        status="COMPLETE",
        started_at_utc=NOW,
        completed_at_utc=NOW + timedelta(minutes=1),
        git=GitReportMetadata(commit_sha="a" * 40, dirty_worktree=False),
        model=ModelReportMetadata(
            artifact_id="yolo11s-ppe",
            version="1",
            family="YOLO11",
            sha256="b" * 64,
            source_url="https://docs.ultralytics.com/models/yolo11/",
            license="AGPL-3.0",
            class_map={str(index): name for index, name in enumerate(CANONICAL_PPE_CLASSES)},
        ),
        dataset=DatasetReportMetadata(
            dataset_id="workersafety25",
            dataset_version="1",
            aggregate_sha256="c" * 64,
            split="test",
            license="CC BY 4.0",
        ),
        runtime=RuntimeReportMetadata(
            python_version="3.12.13",
            platform="Windows-11",
            device="cpu",
            package_versions=package_versions or {"pydantic": "2.13.5", "torch": "2.14.0"},
            hardware=hardware or {"cpu": "test-cpu", "ram": "16 GiB"},
        ),
        thresholds=ThresholdReportMetadata(
            confidence=0.25,
            nms_iou=0.45,
            match_iou=0.5,
        ),
        image_size=ImageSize(width=640, height=640),
        metric_definitions=(
            "Detection match: same class, one-to-one, IoU >= matchIoU",
            "Macro metrics include classes with ground-truth support",
        ),
        command_arguments=(
            CommandArgument(name="datasetManifest", value="D:/data/manifest.json"),
            CommandArgument(name="split", value="test"),
        ),
        detection_metrics=metrics,
        episode_metrics=match_candidate_episodes(
            (), (), onset_tolerance=timedelta(milliseconds=500), evaluated_seconds=60.0
        ),
        accuracy_gate=evaluate_accuracy_gate(metrics),
        failure=None,
    )


def test_accuracy_gate_passes_exact_boundaries() -> None:
    gate = evaluate_accuracy_gate(_detection_metrics())

    assert gate == AccuracyGateResult(status="PASS", reasons=())


def test_accuracy_gate_recomputes_required_values_from_raw_counts() -> None:
    honest = _detection_metrics()
    spoofed_classes = tuple(
        ClassDetectionMetrics(
            class_name=item.class_name,
            true_positives=0,
            false_positives=0,
            false_negatives=40,
            support=40,
            precision=_defined(1.0),
            recall=_defined(1.0),
            f1=_defined(1.0),
        )
        for item in honest.per_class
    )
    spoofed = honest.model_copy(update={"per_class": spoofed_classes})

    gate = evaluate_accuracy_gate(spoofed)

    assert gate.status == "FAIL"
    assert "macro F1 from raw counts 0.0 is below 0.70" in gate.reasons
    assert "NO-Hardhat recall from raw counts 0.0 is below 0.75" in gate.reasons


@pytest.mark.parametrize(
    ("metrics", "reason_fragment"),
    [
        (_detection_metrics(support=29), "support 29 is below 30"),
        (_detection_metrics(macro_f1=0.699999), "macro F1 0.699999 is below 0.70"),
        (
            _detection_metrics(no_hardhat_recall=0.749999),
            "NO-Hardhat recall 0.749999 is below 0.75",
        ),
        (
            _detection_metrics(no_safety_vest_recall=None),
            "NO-Safety Vest recall is undefined",
        ),
        (_detection_metrics(omit_class="Safety Vest"), "missing canonical class: Safety Vest"),
    ],
)
def test_accuracy_gate_fails_with_all_observed_reasons(
    metrics: DetectionMetricsReport, reason_fragment: str
) -> None:
    gate = evaluate_accuracy_gate(metrics)

    assert gate.status == "FAIL"
    assert any(reason_fragment in reason for reason in gate.reasons)


def test_complete_report_requires_all_metadata_metrics_and_consistent_gate() -> None:
    report = _complete_report()
    invalid_payload = report.model_dump()
    invalid_payload["accuracy_gate"] = AccuracyGateResult(status="FAIL", reasons=("fake",))

    assert report.status == "COMPLETE"
    assert report.git.dirty_worktree is False
    with pytest.raises(ValidationError, match="accuracy_gate must be derived"):
        EvaluationReport.model_validate(invalid_payload)


def test_writes_deterministic_sorted_json_with_final_newline(tmp_path: Path) -> None:
    first = _complete_report(
        hardware={"ram": "16 GiB", "cpu": "test-cpu"},
        package_versions={"torch": "2.14.0", "pydantic": "2.13.5"},
    )
    second = _complete_report(
        hardware={"cpu": "test-cpu", "ram": "16 GiB"},
        package_versions={"pydantic": "2.13.5", "torch": "2.14.0"},
    )
    first_path = tmp_path / "first.report.json"
    second_path = tmp_path / "second.report.json"

    write_evaluation_report(first_path, first)
    write_evaluation_report(second_path, second)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")
    raw = first_path.read_text(encoding="utf-8")
    assert raw.index('"accuracyGate"') < raw.index('"commandArguments"')
    payload = json.loads(raw)
    assert payload["schemaVersion"] == "1.0.0"
    assert payload["git"]["dirtyWorktree"] is False


def test_writer_uses_same_directory_temp_and_rejects_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.report.json"
    original_link = os.link
    observed: dict[str, Path] = {}

    def recording_link(source: str | Path, destination: str | Path) -> None:
        observed["source"] = Path(source)
        observed["destination"] = Path(destination)
        original_link(source, destination)

    monkeypatch.setattr("smartsite_ai.evaluation.report.os.link", recording_link)
    write_evaluation_report(path, _complete_report())

    assert observed["source"].parent == path.parent
    assert observed["destination"] == path
    with pytest.raises(EvaluationReportWriteError, match="already exists"):
        write_evaluation_report(path, _complete_report())


def test_publish_failure_removes_temp_and_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.report.json"

    def fail_link(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("smartsite_ai.evaluation.report.os.link", fail_link)

    with pytest.raises(EvaluationReportWriteError, match="could not finalize"):
        write_evaluation_report(path, _complete_report())

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_destination_winner_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.report.json"
    original_link = os.link

    def racing_link(source: str | Path, destination: str | Path) -> None:
        Path(destination).write_bytes(b"concurrent-winner")
        original_link(source, destination)

    monkeypatch.setattr("smartsite_ai.evaluation.report.os.link", racing_link)

    with pytest.raises(EvaluationReportWriteError, match="already exists"):
        write_evaluation_report(path, _complete_report())

    assert path.read_bytes() == b"concurrent-winner"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "forbidden_key",
    ["token", "dbPassword", "serviceCredential", "authorization", "apiKey", "signedUrl"],
)
def test_rejects_sensitive_field_names_recursively(forbidden_key: str) -> None:
    with pytest.raises(ValidationError, match="sensitive field name"):
        _complete_report(hardware={forbidden_key: "must-not-appear"})


def test_incomplete_report_uses_separate_path_and_safe_failure(tmp_path: Path) -> None:
    failure = safe_failure_details(
        RuntimeError(
            "request failed https://admin:secret@example.com?token=raw-secret\nTraceback: hidden"
        )
    )
    report = _complete_report().model_dump()
    report.update(
        status="INCOMPLETE",
        detection_metrics=None,
        episode_metrics=None,
        accuracy_gate=None,
        failure=failure,
    )
    incomplete = EvaluationReport.model_validate(report)
    path = tmp_path / "run.incomplete.json"

    write_evaluation_report(path, incomplete)

    raw = path.read_text(encoding="utf-8")
    assert "raw-secret" not in raw
    assert "admin:secret" not in raw
    assert "Traceback" not in raw
    assert json.loads(raw)["failure"]["errorType"] == "RuntimeError"
    with pytest.raises(EvaluationReportWriteError, match="INCOMPLETE"):
        write_evaluation_report(tmp_path / "bad.report.json", incomplete)


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer supersecret",
        "HTTP 401 with bearer sk-secret",
        "password = my secret value",
        "api_key=top-secret-value",
    ],
)
def test_safe_failure_removes_entire_credential_tail(message: str) -> None:
    failure = safe_failure_details(RuntimeError(message))

    assert "supersecret" not in failure.message
    assert "sk-secret" not in failure.message
    assert "secret value" not in failure.message
    assert "top-secret-value" not in failure.message


@pytest.mark.parametrize("query", ["token=x", "sig=x", "X-Amz-Signature=x", "access_key=x"])
def test_rejects_secret_bearing_source_urls_and_command_values(query: str) -> None:
    with pytest.raises(ValidationError, match="sensitive query parameter"):
        ModelReportMetadata(
            artifact_id="model",
            version="1",
            family="YOLO11",
            sha256="a" * 64,
            source_url=f"https://example.com/model?{query}",
            license="AGPL-3.0",
            class_map={str(i): name for i, name in enumerate(CANONICAL_PPE_CLASSES)},
        )
    with pytest.raises(ValidationError, match="must not contain credentials"):
        CommandArgument(name="header", value="Authorization: Bearer supersecret")
    with pytest.raises(ValidationError, match="must not contain credentials"):
        CommandArgument(name="header", value="Bearer supersecret")


def test_rejects_source_url_fragments_that_can_carry_tokens() -> None:
    with pytest.raises(ValidationError, match="fragment"):
        ModelReportMetadata(
            artifact_id="model",
            version="1",
            family="YOLO11",
            sha256="a" * 64,
            source_url="https://example.com/model#access_token=supersecret",
            license="AGPL-3.0",
            class_map={str(i): name for i, name in enumerate(CANONICAL_PPE_CLASSES)},
        )


def test_rejects_direct_unsafe_failure_and_sensitive_or_duplicate_command_arguments() -> None:
    with pytest.raises(ValidationError, match="failure message contains sensitive data"):
        FailureDetails(error_type="RuntimeError", message="password=hunter2")
    with pytest.raises(ValidationError, match="command argument name"):
        CommandArgument(name="openAIApiKey", value="hidden")

    complete = _complete_report()
    duplicate_arguments = complete.model_dump()
    duplicate_arguments["command_arguments"] = (
        CommandArgument(name="split", value="test"),
        CommandArgument(name="split", value="validation"),
    )
    with pytest.raises(ValidationError, match="command argument names must be unique"):
        EvaluationReport.model_validate(duplicate_arguments)


def test_status_invariants_and_output_suffix_are_enforced(tmp_path: Path) -> None:
    complete = _complete_report()
    with pytest.raises(ValidationError, match="COMPLETE report must not include failure"):
        EvaluationReport.model_validate(
            {**complete.model_dump(), "failure": FailureDetails(error_type="Error", message="x")}
        )
    with pytest.raises(ValidationError, match="INCOMPLETE report requires failure"):
        EvaluationReport.model_validate(
            {
                **complete.model_dump(),
                "status": "INCOMPLETE",
                "detection_metrics": None,
                "episode_metrics": None,
                "accuracy_gate": None,
            }
        )
    with pytest.raises(EvaluationReportWriteError, match="COMPLETE"):
        write_evaluation_report(tmp_path / "bad.incomplete.json", complete)
