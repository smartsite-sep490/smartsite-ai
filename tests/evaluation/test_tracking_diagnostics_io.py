import json
import stat
import subprocess
from pathlib import Path

import pytest

from smartsite_ai.evaluation.tracking_continuity import TrackingMatch
from smartsite_ai.evaluation.tracking_diagnostics_io import (
    TrackingDiagnosticsError,
    load_tracking_diagnostics,
)

SECRET_PERSON = "synthetic-person-secret-key"
MAX_BYTES = 4 * 1024 * 1024
_DOCUMENT = "^tracking diagnostics document is invalid$"
_PATH = "^tracking diagnostics path is not a regular file$"
_SIZE = "^tracking diagnostics file exceeds the size limit$"


def document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "purpose": "TRACKING_ASSOCIATION_DIAGNOSTICS",
        "reviewedBy": "reviewer-1",
        "reviewedAtUtc": "2026-09-30T00:00:00Z",
        "sourceRights": "synthetic attestation only",
        "matches": [match()],
    }
    payload.update(overrides)
    return payload


def match(**overrides: object) -> dict[str, object]:
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


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_maps_reviewed_match_without_treating_it_as_a_corpus(tmp_path: Path) -> None:
    path = write_json(tmp_path / "ledger.json", document())

    loaded = load_tracking_diagnostics(path)

    assert loaded == (
        TrackingMatch(
            camera_id="cam-1",
            clip_id="clip-1",
            session_id="session-1",
            frame_index=0,
            person_key="person-1",
            track_id="track-1",
        ),
    )


def test_empty_matches_still_require_review_metadata(tmp_path: Path) -> None:
    path = write_json(tmp_path / "empty.json", document(matches=[]))

    assert load_tracking_diagnostics(path) == ()


def test_null_predicted_track_is_preserved(tmp_path: Path) -> None:
    path = write_json(tmp_path / "unmatched.json", document(matches=[match(predictedTrackId=None)]))

    assert load_tracking_diagnostics(path)[0].track_id is None


@pytest.mark.parametrize(
    "payload",
    [
        document(schemaVersion="2.0.0"),
        document(purpose="TRACKER_BENCHMARK"),
        document(reviewedBy=" reviewer-1"),
        document(reviewedBy=""),
        document(sourceRights="  rights"),
        document(sourceRights="x" * 1001),
        document(reviewedAtUtc="2026-09-30T00:00:00"),
        document(reviewedAtUtc="2026-09-30T00:00:00+07:00"),
        document(reviewedAtUtc="not-a-date"),
        document(extraField=True),
        {key: value for key, value in document().items() if key != "sourceRights"},
        document(matches=[match(workerId="worker-1")]),
        document(matches=[match(frameIndex=True)]),
        document(matches=[match(frameIndex=-1)]),
        document(matches=[match(frameIndex=2**53)]),
        document(matches=[match(frameIndex=1.5)]),
        document(matches=[match(cameraId=" cam-1")]),
        document(matches=[match(groundTruthPersonKey="")]),
        document(matches=[match(predictedTrackId="")]),
        document(matches=[match(predictedTrackId=" track-1")]),
        document(matches=[match(predictedTrackId=7)]),
        document(matches="not-an-array"),
    ],
)
def test_loader_rejects_strict_schema_violations(tmp_path: Path, payload: object) -> None:
    path = write_json(tmp_path / "bad.json", payload)

    with pytest.raises(TrackingDiagnosticsError, match=_DOCUMENT):
        load_tracking_diagnostics(path)


def test_loader_rejects_duplicate_keys_nan_and_malformed_text(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schemaVersion":"1.0.0","schemaVersion":"1.0.0","purpose":"TRACKING_ASSOCIATION_DIAGNOSTICS",'
        '"reviewedBy":"reviewer-1","reviewedAtUtc":"2026-09-30T00:00:00Z",'
        '"sourceRights":"synthetic","matches":[]}',
        encoding="utf-8",
    )
    nested = tmp_path / "nested.json"
    nested.write_text(
        '{"schemaVersion":"1.0.0","purpose":"TRACKING_ASSOCIATION_DIAGNOSTICS",'
        '"reviewedBy":"reviewer-1","reviewedAtUtc":"2026-09-30T00:00:00Z",'
        '"sourceRights":"synthetic","matches":[{"cameraId":"cam-1","cameraId":"cam-1",'
        '"clipId":"clip-1","streamSessionId":"session-1","frameIndex":0,'
        '"groundTruthPersonKey":"person-1","predictedTrackId":null}]}',
        encoding="utf-8",
    )
    nan_file = tmp_path / "nan.json"
    nan_file.write_text(
        json.dumps(document()).replace('"frameIndex": 0', '"frameIndex": NaN'),
        encoding="utf-8",
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")

    for path in (duplicate, nested, nan_file, malformed, invalid_utf8):
        with pytest.raises(TrackingDiagnosticsError, match=_DOCUMENT):
            load_tracking_diagnostics(path)


def test_loader_error_does_not_echo_secret_identifiers(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "secret.json",
        document(matches=[match(groundTruthPersonKey=SECRET_PERSON, frameIndex=True)]),
    )

    with pytest.raises(TrackingDiagnosticsError) as caught:
        load_tracking_diagnostics(path)

    message = str(caught.value)
    assert SECRET_PERSON not in message
    assert path.name not in message
    assert str(tmp_path) not in message


def test_loader_rejects_directory_missing_and_oversize_paths(tmp_path: Path) -> None:
    write_json(tmp_path / "ok.json", document(matches=[]))
    huge = tmp_path / "huge.json"
    huge.write_bytes(b"{" + b" " * MAX_BYTES)

    with pytest.raises(TrackingDiagnosticsError, match=_PATH):
        load_tracking_diagnostics(tmp_path)
    with pytest.raises(TrackingDiagnosticsError, match=_PATH):
        load_tracking_diagnostics(tmp_path / "missing.json")
    with pytest.raises(TrackingDiagnosticsError, match=_SIZE):
        load_tracking_diagnostics(huge)
    assert huge.stat().st_size > MAX_BYTES


def test_loader_rejects_symlink_before_reading_target(tmp_path: Path) -> None:
    target = write_json(
        tmp_path / "target.json",
        document(matches=[match(groundTruthPersonKey=SECRET_PERSON)]),
    )
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    with pytest.raises(TrackingDiagnosticsError, match=_PATH) as caught:
        load_tracking_diagnostics(link)

    assert SECRET_PERSON not in str(caught.value)


def test_loader_rejects_reparse_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    write_json(real_dir / "ledger.json", document(matches=[]))
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real_dir)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not junction.exists():
        pytest.skip("directory junctions are unavailable")

    with pytest.raises(TrackingDiagnosticsError, match=_PATH):
        load_tracking_diagnostics(junction / "ledger.json")
    attributes = junction.lstat().st_file_attributes
    assert attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT


def test_loader_accepts_maximum_safe_frame_and_text_bounds(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "bounds.json",
        document(
            reviewedBy="r" * 128,
            sourceRights="s" * 1000,
            matches=[match(frameIndex=2**53 - 1, predictedTrackId=None)],
        ),
    )

    loaded = load_tracking_diagnostics(path)

    assert loaded[0].frame_index == 2**53 - 1
    assert loaded[0].track_id is None


def test_match_array_limit_is_enforced_under_the_file_budget(tmp_path: Path) -> None:
    rows = [
        match(frameIndex=index, groundTruthPersonKey=f"p{index}", predictedTrackId=f"t{index}")
        for index in range(20_001)
    ]
    path = write_json(tmp_path / "too-many.json", document(matches=rows))
    assert path.stat().st_size <= MAX_BYTES

    with pytest.raises(TrackingDiagnosticsError, match=_DOCUMENT):
        load_tracking_diagnostics(path)


def test_infinity_constant_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "infinity.json"
    path.write_text(
        json.dumps(document()).replace('"frameIndex": 0', '"frameIndex": Infinity'),
        encoding="utf-8",
    )

    with pytest.raises(TrackingDiagnosticsError, match=_DOCUMENT):
        load_tracking_diagnostics(path)


def test_loader_normalizes_deep_json_nesting(tmp_path: Path) -> None:
    path = tmp_path / "secret-ledger.json"
    path.write_text("[" * 5_000 + f'"{SECRET_PERSON}"' + "]" * 5_000, encoding="utf-8")

    with pytest.raises(TrackingDiagnosticsError, match=_DOCUMENT) as caught:
        load_tracking_diagnostics(path)

    message = str(caught.value)
    assert SECRET_PERSON not in message
    assert path.name not in message
    assert str(tmp_path) not in message
    assert caught.value.__cause__ is None


def test_loader_normalizes_invalid_local_path(tmp_path: Path) -> None:
    path = tmp_path / "secret-ledger\x00.json"

    with pytest.raises(TrackingDiagnosticsError, match=_PATH) as caught:
        load_tracking_diagnostics(path)

    message = str(caught.value)
    assert "secret-ledger" not in message
    assert str(tmp_path) not in message
    assert caught.value.__cause__ is None
