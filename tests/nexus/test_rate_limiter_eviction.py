"""Regression tests for lead L-02: rate-limiter state must be bounded.

The in-process fixed-window limiter previously kept one bucket per identity
forever.  These tests pin the designed behaviour:

- a hard identity cap bounds the bucket dictionary;
- expired buckets are swept first (least destructive eviction);
- when nothing is expired, eviction falls back to lowest-count-then-oldest
  so an active victim (high count) survives an identity flood;
- 429 / Retry-After semantics and concurrency safety are preserved.
"""

from __future__ import annotations

import threading

import pytest

from nexus.api.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


def test_identity_cap_bounds_bucket_dictionary(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60, clock=clock, max_identities=50)
    for i in range(200):
        limiter.check(f"id{i}")
    assert len(limiter._buckets) <= 50


def test_expired_buckets_are_swept(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60, clock=clock, max_identities=50)
    for i in range(50):
        limiter.check(f"id{i}")
    assert len(limiter._buckets) == 50
    clock.now += 61  # everything expires
    for i in range(50):
        limiter.check(f"new{i}")
    assert len(limiter._buckets) <= 50


def test_active_victim_survives_flood(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=100, window_seconds=60, clock=clock, max_identities=50)
    for _ in range(10):
        limiter.check("victim")           # victim is active, count -> 10
    for i in range(60):                    # flood beyond the cap within the window
        limiter.check(f"attacker{i}")
    allowed, _ = limiter.check("victim")  # victim must still be tracked, not evicted
    assert allowed


def test_retry_after_semantics(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60, clock=clock)
    assert limiter.check("x")[0] is True
    assert limiter.check("x")[0] is True
    allowed, retry_after = limiter.check("x")
    assert allowed is False
    assert retry_after >= 1 and retry_after <= 61
    clock.now += 61
    assert limiter.check("x")[0] is True


def test_concurrent_checks_allow_exactly_max(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=100, window_seconds=60, clock=clock)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(50):
            allowed, _ = limiter.check("same")
            with lock:
                results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 100
    assert all(not r for r in results[100:])


def test_reset_clears_state(clock: FakeClock) -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=60, clock=clock)
    limiter.check("x")
    limiter.check("y")
    limiter.reset()
    assert len(limiter._buckets) == 0
