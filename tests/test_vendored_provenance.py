import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMARTSITE_MERGE_SHA = "691b7acc9f62bf72e30805a8c0ccb3836f15ac3d"


def test_vendored_schema_matches_real_provenance():
    metadata = json.loads((ROOT / "contracts/metadata.json").read_text(encoding="utf-8"))
    schema_bytes = (ROOT / "contracts/schemas/v1/technical-observation-event.json").read_bytes()
    golden_bytes = (ROOT / "contracts/golden-vectors.json").read_bytes()
    region_schema_bytes = (
        ROOT / "contracts/schemas/v1/camera-region-configuration.json"
    ).read_bytes()
    region_vectors_bytes = (
        ROOT / "contracts/camera-region-configuration-vectors.json"
    ).read_bytes()
    assert metadata["sourceRepository"] == "smartsite-sep490/smartsite"
    assert metadata["sourceCommitSha"] == SMARTSITE_MERGE_SHA
    assert metadata["schemaVersion"] == "1.0.0"
    assert metadata["schemaSha256"] == hashlib.sha256(schema_bytes).hexdigest()
    assert metadata["goldenVectorsSha256"] == hashlib.sha256(golden_bytes).hexdigest()
    region_metadata = metadata["cameraRegionConfiguration"]
    assert region_metadata["schemaVersion"] == "1.0.0"
    assert re.fullmatch(r"[0-9a-f]{64}", region_metadata["schemaSha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", region_metadata["goldenVectorsSha256"])
    assert region_metadata["schemaSha256"] == hashlib.sha256(region_schema_bytes).hexdigest()
    assert (
        region_metadata["goldenVectorsSha256"] == hashlib.sha256(region_vectors_bytes).hexdigest()
    )
