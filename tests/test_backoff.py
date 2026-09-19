from smartsite_ai.ingestion.backoff import ExponentialBackoff


def test_exponential_backoff_deterministic_progression() -> None:
    backoff = ExponentialBackoff(
        initial_delay=1.0,
        max_delay=10.0,
        factor=2.0,
        jitter=0.0,
    )

    assert backoff.next_delay(attempt=0) == 1.0
    assert backoff.next_delay(attempt=1) == 2.0
    assert backoff.next_delay(attempt=2) == 4.0
    assert backoff.next_delay(attempt=3) == 8.0
    assert backoff.next_delay(attempt=4) == 10.0
    assert backoff.next_delay(attempt=5) == 10.0


def test_exponential_backoff_jitter_bounds() -> None:
    backoff = ExponentialBackoff(
        initial_delay=2.0,
        max_delay=20.0,
        factor=2.0,
        jitter=0.2,  # +/- 20%
    )

    # For attempt 0, nominal is 2.0 -> bounds [1.6, 2.4]
    for _ in range(50):
        d = backoff.next_delay(attempt=0)
        assert 1.6 <= d <= 2.4

    # For attempt 1, nominal is 4.0 -> bounds [3.2, 4.8]
    for _ in range(50):
        d = backoff.next_delay(attempt=1)
        assert 3.2 <= d <= 4.8


def test_exponential_backoff_stateful_compute() -> None:
    backoff = ExponentialBackoff(
        initial_delay=0.5,
        max_delay=5.0,
        factor=2.0,
        jitter=0.0,
    )

    assert backoff.compute_next_delay() == 0.5  # attempt 0
    assert backoff.attempts == 1
    assert backoff.compute_next_delay() == 1.0  # attempt 1
    assert backoff.attempts == 2
    assert backoff.compute_next_delay() == 2.0  # attempt 2
    assert backoff.attempts == 3

    backoff.reset()
    assert backoff.attempts == 0
    assert backoff.compute_next_delay() == 0.5
