import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_schema_matches_real_provenance():
    metadata = json.loads((ROOT / "contracts/metadata.json").read_text(encoding="utf-8"))
    schema_bytes = (ROOT / "contracts/schemas/v1/technical-observation-event.json").read_bytes()
    golden_bytes = (ROOT / "contracts/golden-vectors.json").read_bytes()
    assert metadata["sourceRepository"] == "smartsite-sep490/smartsite"
    assert re.fullmatch(r"[0-9a-f]{40}", metadata["sourceCommitSha"])
    assert metadata["schemaVersion"] == "1.0.0"
    assert metadata["schemaSha256"] == hashlib.sha256(schema_bytes).hexdigest()
    assert metadata["goldenVectorsSha256"] == hashlib.sha256(golden_bytes).hexdigest()
