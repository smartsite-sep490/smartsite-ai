import hashlib
import json
from pathlib import Path

from smartsite_ai.domain.policy_configuration_bundle import CameraPolicyConfigurationBundle

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMIT = "0a690f745091923de55c20c2765c99074c5f1d36"


def test_policy_schema_and_example_keep_immutable_canonical_provenance():
    metadata = json.loads((ROOT / "contracts/metadata.json").read_text(encoding="utf-8"))
    policy = metadata["cameraPolicyConfigurationBundle"]
    assert policy["sourceRepository"] == "smartsite-sep490/smartsite"
    assert policy["sourceCommitSha"] == CANONICAL_COMMIT
    assert policy["schemaVersion"] == "2.0.0"
    for path, digest in (
        (policy["path"], policy["schemaSha256"]),
        (policy["examplePath"], policy["exampleSha256"]),
    ):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert policy["path"] == policy["sourcePath"]
    assert policy["examplePath"] == policy["sourceExamplePath"]

    # The immutable schema uses ECMAScript Unicode patterns (\p{}), which Python
    # stdlib re cannot execute. Paired TypeScript/Python checks cover semantics;
    # this check proves immutable bytes/provenance and the canonical wire example.
    wire = (ROOT / policy["examplePath"]).read_bytes()
    original = json.loads(wire)
    decoded = CameraPolicyConfigurationBundle.from_wire_bytes(wire)
    assert decoded.bundle_sha256 == original["bundleSha256"]
    assert decoded.model_dump(mode="json", by_alias=True) == original
