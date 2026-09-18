import json
from pathlib import Path

import pytest

from smartsite_ai.core.canonical_hash import compute_canonical_payload_hash

ROOT = Path(__file__).resolve().parents[1]


def test_python_jcs_matches_typescript_golden_vectors():
    vectors = json.loads((ROOT / "contracts/golden-vectors.json").read_text(encoding="utf-8"))
    assert len(vectors) == 3
    for vector in vectors:
        assert compute_canonical_payload_hash(vector["input"]) == vector["expectedSha256"]


def test_python_jcs_rejects_invalid_payloads():
    with pytest.raises(ValueError, match="Payload cannot be canonicalized"):
        # jcs cannot canonicalize non-serializable objects
        compute_canonical_payload_hash(object())
