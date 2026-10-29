"""Bounded metadata retention, independent of model/network execution."""

import pytest

from smartsite_ai import mf06_identity_cache as module
from smartsite_ai.mf06_identity_cache import Mf06IdentityCache


def candidate():
    return {"state": "CANDIDATE", "profileReferenceHash": "a" * 64, "profileRevision": "revision"}


def test_cache_replays_copies_and_rejects_changed_input():
    cache = Mf06IdentityCache(object())
    response = candidate()
    cache.remember("invocation", "request-hash", response)
    response["profileReferenceHash"] = "changed"
    first = cache.lookup("invocation", "request-hash")
    assert first == candidate()
    first["profileReferenceHash"] = "also changed"
    assert cache.lookup("invocation", "request-hash") == candidate()
    with pytest.raises(ValueError, match="MF06_INVOCATION_CONFLICT"):
        cache.lookup("invocation", "different-hash")


def test_cache_expiry_is_monotonic_and_replay_does_not_extend_retention(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    cache = Mf06IdentityCache(object())
    cache.remember("invocation", "hash", candidate())
    now[0] = 59.0
    assert cache.lookup("invocation", "hash") == candidate()
    now[0] = 60.0
    assert cache.lookup("invocation", "hash") is None


def test_cache_evicts_least_recent_entry_at_fixed_bound(monkeypatch):
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    cache = Mf06IdentityCache(object())
    for key in ("a", "b"):
        cache.remember(key, key, candidate())
    assert cache.lookup("a", "a") == candidate()
    cache.remember("c", "c", candidate())
    assert cache.lookup("b", "b") is None
    assert cache.lookup("a", "a") == candidate()
    assert cache.lookup("c", "c") == candidate()


def test_cache_retains_no_transient_failures_and_close_prevents_late_repopulation():
    cache = Mf06IdentityCache(object())
    for reason in ("SERVICE_UNAVAILABLE", "ABORTED"):
        cache.remember("invocation", "hash", {"state": "UNAVAILABLE", "reason": reason})
        assert cache.lookup("invocation", "hash") is None
    no_match = {"state": "UNAVAILABLE", "reason": "NO_MATCH"}
    cache.remember("invocation", "hash", no_match)
    assert cache.lookup("invocation", "hash") == no_match
    cache.close()
    assert cache.provider is None
    cache.remember("invocation", "hash", candidate())
    assert cache.lookup("invocation", "hash") is None
