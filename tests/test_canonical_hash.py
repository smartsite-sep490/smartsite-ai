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


@pytest.mark.parametrize("unsafe", [2**53, 2**53 + 1, -(2**53)])
def test_python_jcs_rejects_integers_outside_interoperable_domain(unsafe: int):
    with pytest.raises(ValueError, match="safe integer"):
        compute_canonical_payload_hash({"trackId": unsafe})


def test_python_jcs_matches_rfc_8785_primitive_serialization_example():
    payload = {
        "numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27],
        "string": '€$\u000f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }
    expected = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    import jcs

    assert jcs.canonicalize(payload).decode("utf-8") == expected
