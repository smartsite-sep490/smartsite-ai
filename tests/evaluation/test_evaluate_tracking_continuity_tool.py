import json
import os
import subprocess
from pathlib import Path

PYTHON = Path(r"D:\Ky9-FPT\SmartSite\repos\smartsite-ai\.venv\Scripts\python.exe")
SRC = Path(r"D:\Ky9-FPT\SmartSite\worktrees\evaluation-input-tooling\src")
SECRET_PERSON = "synthetic-person-secret-key"


def ledger(matches: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schemaVersion": "1.0.0",
        "purpose": "TRACKING_ASSOCIATION_DIAGNOSTICS",
        "reviewedBy": "reviewer-1",
        "reviewedAtUtc": "2026-09-30T00:00:00Z",
        "sourceRights": "synthetic attestation only",
        "matches": matches,
    }


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "cameraId": "cam-1",
        "clipId": "clip-1",
        "streamSessionId": "session-1",
        "frameIndex": 0,
        "groundTruthPersonKey": "person-1",
        "predictedTrackId": "track-1",
    }
    payload.update(overrides)
    return payload


def run_cli(
    tmp_path: Path,
    payload: object,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "secret-ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    arguments = [str(PYTHON), "-m", "smartsite_ai.tools.evaluate_tracking_continuity"]
    if extra_args is None:
        arguments.extend(["--input", str(path)])
    else:
        arguments.extend(extra_args)
    return subprocess.run(
        arguments,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_prints_matched_ledger_report_and_writes_no_file(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())

    completed = run_cli(tmp_path, ledger([row(), row(frameIndex=1, predictedTrackId="track-1")]))

    assert completed.returncode == 0
    report = json.loads(completed.stdout)
    assert report["scope"] == "MATCHED_LEDGER_DIAGNOSTICS"
    assert "idf1" not in completed.stdout.lower()
    assert "hota" not in completed.stdout.lower()
    assert set(tmp_path.iterdir()) == before | {tmp_path / "secret-ledger.json"}
    assert completed.stderr == ""


def test_cli_invalid_input_exits_2_without_payload_or_path(tmp_path: Path) -> None:
    completed = run_cli(
        tmp_path,
        ledger([row(groundTruthPersonKey=SECRET_PERSON, frameIndex=True)]),
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.strip() == "tracking diagnostics input rejected"
    assert SECRET_PERSON not in completed.stderr
    assert "secret-ledger.json" not in completed.stderr
    assert str(tmp_path) not in completed.stderr


def test_cli_rejects_missing_arguments_without_usage_leak(tmp_path: Path) -> None:
    completed = run_cli(tmp_path, ledger([]), extra_args=[])

    assert completed.returncode == 2
    assert completed.stderr.strip() == "tracking diagnostics input rejected"
    assert "usage" not in completed.stderr.lower()
