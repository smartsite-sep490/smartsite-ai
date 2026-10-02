import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartsite_ai.evaluation.split_audit import SplitAuditManifest, audit_split_groups
from smartsite_ai.tools.audit_dataset_splits import run


def sample(name: str, split: str, digest: str, group: str | None = None) -> dict:
    return {"sampleId": name, "split": split, "sha256": digest * 64, "sourceGroupId": group}


def run_audit(tmp_path: Path, samples: list[dict], links: list[dict] | None = None):
    source = tmp_path / "source.json"
    output = tmp_path / "audit.json"
    source.write_text(
        json.dumps({"schemaVersion": "1.0.0", "samples": samples, "links": links or []})
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsite_ai.tools.audit_dataset_splits",
            "--manifest",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(output.read_text()) if output.exists() else None
    return result, report


def test_independent_declared_groups_are_not_model_acceptance(tmp_path: Path) -> None:
    result, report = run_audit(
        tmp_path,
        [
            sample("a", "train", "a", "scene-a"),
            sample("b", "val", "b", "scene-b"),
            sample("c", "test", "c", "scene-c"),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert report["status"] == "NO_KNOWN_OVERLAP"
    assert report["datasetAccepted"] is False
    assert report["confirmedCrossSplitGroups"] == []


@pytest.mark.parametrize("same_group,same_hash", [(True, False), (False, True)])
def test_known_source_or_byte_identity_across_splits_blocks(
    tmp_path: Path, same_group: bool, same_hash: bool
) -> None:
    result, report = run_audit(
        tmp_path,
        [
            sample("a", "train", "a", "scene-a"),
            sample("b", "test", "a" if same_hash else "b", "scene-a" if same_group else "scene-b"),
        ],
    )
    assert result.returncode == 2, result.stderr
    assert report["status"] == "BLOCKED_KNOWN_OVERLAP"
    assert report["confirmedCrossSplitGroups"][0]["sampleIds"] == ["a", "b"]


def test_candidate_chain_requires_review_without_claiming_confirmed_duplicate(
    tmp_path: Path,
) -> None:
    result, report = run_audit(
        tmp_path,
        [sample("a", "train", "a"), sample("b", "train", "b"), sample("c", "test", "c")],
        [
            {"leftSampleId": "a", "rightSampleId": "b", "reason": "phash-candidate"},
            {"leftSampleId": "b", "rightSampleId": "c", "reason": "phash-candidate"},
        ],
    )
    assert result.returncode == 2, result.stderr
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["confirmedCrossSplitGroups"] == []
    assert report["candidateCrossSplitGroups"][0]["sampleIds"] == ["a", "b", "c"]
    assert report["datasetAccepted"] is False


def test_missing_source_groups_cannot_claim_independent_holdout(tmp_path: Path) -> None:
    result, report = run_audit(tmp_path, [sample("a", "train", "a"), sample("b", "test", "b")])
    assert result.returncode == 2, result.stderr
    assert report["status"] == "INCOMPLETE_SOURCE_GROUPS"
    assert report["samplesWithoutSourceGroup"] == ["a", "b"]


@pytest.mark.parametrize(
    "field,value",
    [("sampleId", " a"), ("sourceGroupId", " "), ("sourceGroupId", "scene\nA")],
)
def test_ambiguous_identifiers_are_rejected_without_silent_normalization(
    tmp_path: Path, field: str, value: str
) -> None:
    rows = [sample("a", "train", "a", "scene-a")]
    rows[0][field] = value
    result, report = run_audit(tmp_path, rows)
    assert result.returncode == 1
    assert "Dataset split audit rejected:" in result.stderr
    assert report is None


@pytest.mark.parametrize(
    "broken", ["duplicate-id", "missing-link-target", "bad-sha", "unknown-field", "unknown-split"]
)
def test_malformed_manifest_is_rejected_without_publishing(tmp_path: Path, broken: str) -> None:
    samples = [sample("a", "train", "a", "scene-a"), sample("b", "test", "b", "scene-b")]
    links = []
    if broken == "duplicate-id":
        samples[1]["sampleId"] = "a"
    elif broken == "missing-link-target":
        links = [{"leftSampleId": "a", "rightSampleId": "missing", "reason": "phash"}]
    elif broken == "bad-sha":
        samples[0]["sha256"] = "not-sha"
    elif broken == "unknown-field":
        samples[0]["accepted"] = True
    else:
        samples[0]["split"] = "holdout"
    result, report = run_audit(tmp_path, samples, links)
    assert result.returncode == 1
    assert "Dataset split audit rejected:" in result.stderr
    assert report is None


def test_existing_report_and_manifest_are_never_overwritten(tmp_path: Path) -> None:
    result, _ = run_audit(
        tmp_path, [sample("a", "train", "a", "scene-a"), sample("b", "test", "b", "scene-b")]
    )
    assert result.returncode == 0, result.stderr
    before = (tmp_path / "audit.json").read_bytes()
    result, _ = run_audit(
        tmp_path, [sample("a", "train", "a", "scene-a"), sample("b", "test", "b", "scene-b")]
    )
    assert result.returncode == 1
    assert (tmp_path / "audit.json").read_bytes() == before


def test_report_is_deterministic_and_candidate_edges_do_not_become_confirmed() -> None:
    samples = [
        sample("a", "train", "a", "scene-a"),
        sample("b", "train", "b", "scene-a"),
        sample("c", "test", "c", "scene-c"),
    ]
    links = [{"leftSampleId": "b", "rightSampleId": "c", "reason": "similar-candidate"}]
    first = audit_split_groups(
        SplitAuditManifest.model_validate(
            {"schemaVersion": "1.0.0", "samples": samples, "links": links}
        )
    )
    second = audit_split_groups(
        SplitAuditManifest.model_validate(
            {
                "schemaVersion": "1.0.0",
                "samples": list(reversed(samples)),
                "links": [
                    {"leftSampleId": "c", "rightSampleId": "b", "reason": "similar-candidate"}
                ],
            }
        )
    )
    assert first == second
    assert first["confirmedCrossSplitGroups"] == []
    assert first["candidateCrossSplitGroups"][0]["sampleIds"] == ["a", "b", "c"]
    assert first["fileBytesVerified"] is False


def test_report_identifies_missing_split_without_claiming_acceptance() -> None:
    manifest = SplitAuditManifest.model_validate(
        {"schemaVersion": "1.0.0", "samples": [sample("a", "train", "a", "scene-a")], "links": []}
    )
    report = audit_split_groups(manifest)
    assert report["missingSplits"] == ["val", "test"]
    assert report["datasetAccepted"] is False


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"schemaVersion":"1.0.0","samples":[],"samples":[]}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run(source, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_size_bound_rejects_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smartsite_ai.tools.audit_dataset_splits as tool

    source = tmp_path / "source.json"
    source.write_text("a" * 50)
    monkeypatch.setattr(tool, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(ValueError, match="manifest exceeds"):
        run(source, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_concurrent_writer_is_preserved_and_owned_temp_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smartsite_ai.tools.audit_dataset_splits as tool

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "samples": [sample("a", "train", "a", "scene-a")],
                "links": [],
            }
        )
    )
    output = tmp_path / "audit.json"
    real_link = tool.os.link

    def racing_link(a, b):
        output.write_text("other writer")
        return real_link(a, b)

    monkeypatch.setattr(tool.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        run(source, output)
    assert output.read_text() == "other writer"
    assert list(tmp_path.glob(".split-audit-*")) == []
