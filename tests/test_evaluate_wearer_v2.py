import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smartsite_ai.tools.evaluate_wearer_v2 import MAX_INPUT_BYTES, main


def payload():
    return {
        "schema_version": "2.0.0",
        "reference_origin": "SYNTHETIC",
        "references": [
            {
                "clip_id": "synthetic-clip",
                "frame_index": 1,
                "person_instance_id": "review-person-1",
                "item": "GLOVES",
                "status": "MISSING",
            }
        ],
        "predictions": [],
    }


def test_cli_writes_explicit_unknown_coverage_report(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(payload()), encoding="utf-8")
    assert main(["--input", str(source), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    gloves = next(row for row in result["items"] if row["item"] == "GLOVES")
    assert gloves["abstained_observable"] == 1
    assert gloves["missing_recall"]["value"] == 0.0
    assert result["schema_version"] == "2.1.0"
    assert gloves["present_recall"]["reason"] == "NO_VISIBLE_PRESENT_REFERENCE"
    assert result["model_accepted"] is False


@pytest.mark.parametrize("change", ["version", "unknown_field", "bad_json"])
def test_cli_refuses_bad_inputs_without_creating_report(tmp_path, capsys, change):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    values = payload()
    if change == "version":
        values["schema_version"] = "1.0.0"
    if change == "unknown_field":
        values["secret"] = "synthetic-secret-must-not-be-logged"
    source.write_text("{" if change == "bad_json" else json.dumps(values), encoding="utf-8")
    assert main(["--input", str(source), "--output", str(output)]) == 2
    assert not output.exists()
    assert "synthetic-secret" not in capsys.readouterr().err


def test_cli_refuses_oversize_payload_and_preserves_existing_output(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
    assert main(["--input", str(source), "--output", str(output)]) == 2
    assert not output.exists()
    source.write_text(json.dumps(payload()), encoding="utf-8")
    output.write_text("existing-proof", encoding="utf-8")
    assert main(["--input", str(source), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "existing-proof"


def test_cli_rejects_conflicting_duplicate_json_fields(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    text = json.dumps(payload()).replace(
        '"status": "MISSING"', '"status": "PRESENT", "status": "MISSING"'
    )
    source.write_text(text, encoding="utf-8")
    assert main(["--input", str(source), "--output", str(output)]) == 2
    assert not output.exists()


def test_clean_import_and_actual_cli_never_import_vision_runtimes(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(payload()), encoding="utf-8")
    code = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cv2', 'ultralytics'}:
        raise AssertionError('forbidden vision import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from smartsite_ai.tools.evaluate_wearer_v2 import main
assert main(sys.argv[1:]) == 0
assert not any(name.split('.')[0] in {'torch', 'cv2', 'ultralytics'} for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", code, "--input", str(source), "--output", str(output)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["model_accepted"] is False


def test_cli_refuses_nonregular_input_before_attempting_read(tmp_path, monkeypatch):
    import stat
    from types import SimpleNamespace

    source = tmp_path / "synthetic-fifo"
    output = tmp_path / "report.json"
    real_stat = Path.stat
    real_open = Path.open

    def synthetic_stat(path, *args, **kwargs):
        if path == source:
            return SimpleNamespace(st_mode=stat.S_IFIFO)
        return real_stat(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if path == source:
            raise AssertionError("a nonregular input must not be opened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", synthetic_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    assert main(["--input", str(source), "--output", str(output)]) == 2
    assert not output.exists()
