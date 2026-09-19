import hashlib
from typing import Any

import jcs

MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _assert_interoperable_integers(payload: Any, seen: set[int] | None = None) -> None:
    if isinstance(payload, bool) or payload is None:
        return
    if isinstance(payload, int) and abs(payload) > MAX_SAFE_INTEGER:
        raise ValueError(f"Integer {payload} is outside the interoperable JSON safe integer domain")
    if isinstance(payload, float) and payload.is_integer() and abs(payload) > MAX_SAFE_INTEGER:
        raise ValueError(f"Integer {payload} is outside the interoperable JSON safe integer domain")

    if not isinstance(payload, (dict, list, tuple)):
        return

    seen = seen if seen is not None else set()
    identity = id(payload)
    if identity in seen:
        return
    seen.add(identity)

    values = payload.values() if isinstance(payload, dict) else payload
    for value in values:
        _assert_interoperable_integers(value, seen)


def compute_canonical_payload_hash(payload: Any) -> str:
    """Computes the deterministic SHA-256 hash of an RFC 8785 canonicalized JSON payload.

    Matches canonicalize in @smartsite/contracts exactly.
    """
    _assert_interoperable_integers(payload)

    try:
        canonical = jcs.canonicalize(payload)
    except Exception as exc:
        raise ValueError("Payload cannot be canonicalized as RFC 8785 JSON") from exc

    if canonical is None:
        raise ValueError("Payload cannot be canonicalized as RFC 8785 JSON")

    return hashlib.sha256(canonical).hexdigest()
