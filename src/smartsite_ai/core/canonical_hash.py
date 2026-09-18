import hashlib
from typing import Any

import jcs


def compute_canonical_payload_hash(payload: Any) -> str:
    """Computes the deterministic SHA-256 hash of an RFC 8785 canonicalized JSON payload.

    Matches canonicalize in @smartsite/contracts exactly.
    """
    try:
        canonical = jcs.canonicalize(payload)
    except Exception as exc:
        raise ValueError("Payload cannot be canonicalized as RFC 8785 JSON") from exc

    if canonical is None:
        raise ValueError("Payload cannot be canonicalized as RFC 8785 JSON")

    return hashlib.sha256(canonical).hexdigest()
