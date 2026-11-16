import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMIT = "cf964554dcc45ba34a0161fcb1d0fe48485fb828"


def test_tracked_contract_versions_have_separate_immutable_provenance():
    metadata = json.loads((ROOT / "contracts/metadata.json").read_text())
    entries = metadata["trackedTechnicalObservationVersions"]
    assert [entry["schemaVersion"] for entry in entries] == ["1.5.0", "1.6.0", "1.7.0"]
    for entry in entries:
        assert entry["sourceRepository"] == "smartsite-sep490/smartsite"
        assert entry["sourceCommitSha"] == CANONICAL_COMMIT
        assert entry["path"] == entry["sourcePath"]
        assert (
            hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["schemaSha256"]
        )
    example = entries[-1]
    assert example["examplePath"] == example["sourceExamplePath"]
    assert (
        hashlib.sha256((ROOT / example["examplePath"]).read_bytes()).hexdigest()
        == example["exampleSha256"]
    )
    # Vendoring tracked versions never silently widens the old wire DTO/profile.
    assert metadata["technicalObservationVersions"][-1]["schemaVersion"] == "1.4.0"
