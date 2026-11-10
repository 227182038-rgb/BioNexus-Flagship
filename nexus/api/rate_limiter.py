"""In-process fixed-window rate limiter for the BioNexus API.

This module provides a lightweight, thread-safe rate limiter that tracks
request counts per identity within fixed time windows.

**Important limitation**: This is an IN-PROCESS limiter.  It is suitable for
single-process deployments, development, and small deployments.  It is NOT
a distributed or global limiter — each application worker maintains its own
independent state.  For multi-worker or multi-server deployments, use an
external rate-limiting layer (e.g. API gateway, Redis-backed limiter).

Identity resolution:
- When API key authentication is configured, the identity is a SHA-256
  fingerprint of the authenticated key (never the raw key itself).
- When authentication is disabled (dev/test mode), the identity is the
  client IP address.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Callable


class RateLimiter:
    """Fixed-window in-process rate limiter.

    Parameters
    ----------
    max_requests:
        Maximum number of requests permitted per window per identity.
    window_seconds:
        Duration of each rate-limit window in seconds.
    clock:
        Callable that returns the current time as a float (seconds since
        epoch).  Defaults to ``time.monotonic`` but can be replaced in
        tests for deterministic behaviour without sleeps.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        *,
        clock: Callable[[], float] | None = None,
        max_identities: int = 10000,
    ) -> None:
        import time as _time

        if max_identities < 1:
            raise ValueError("max_identities must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock or _time.monotonic
        self._max_identities = max_identities
        # _buckets maps identity → [window_start, count]
        self._buckets: dict[str, list[float | int]] = {}
        self._lock = threading.Lock()

    @property
    def max_requests(self) -> int:
        return self._max_requests

    @property
    def window_seconds(self) -> int:
        return self._window_seconds

    @property
    def max_identities(self) -> int:
        """Hard cap on tracked identities (memory bound)."""
        return self._max_identities

    def check(self, identity: str) -> tuple[bool, int]:
        """Check whether *identity* may proceed.

        Returns ``(allowed, retry_after)`` where *allowed* is ``True``
        when the request is within the configured limit and *retry_after*
        is the number of whole seconds until the current window resets
        (meaningful only when ``allowed`` is ``False``).
        """
        now = self._clock()
        with self._lock:
            self._evict_if_over_cap(now)
            bucket = self._buckets.get(identity)
            if bucket is None or now - bucket[0] >= self._window_seconds:
                # New window — reset
                self._buckets[identity] = [now, 1]
                return True, 0
            # Existing window
            bucket[1] += 1  # type: ignore[operator]
            if bucket[1] <= self._max_requests:
                return True, 0
            retry_after = int(self._window_seconds - (now - bucket[0])) + 1
            return False, max(retry_after, 1)

    def _evict_if_over_cap(self, now: float) -> None:
        """Keep the bucket dictionary bounded (called under ``self._lock``).

        Eviction only occurs when a NEW identity must be inserted while the
        dictionary is at capacity:

        1. Expired windows are dropped first — they carry no retained state
           and would reset on next use anyway.
        2. If nothing is expired, the identity with the *lowest request count*
           (ties: oldest window) is evicted.  This keeps a victim that is
           actively sending requests (higher count) from being flushed by an
           attacker who merely mints many idle identities; identity floods
           that hold counts near 1 instead end up recycling themselves.

        An identity whose bucket already exists is never evicted by its own
        check.  Documented limitation: a determined identity flood can still
        force eviction of *idle* low-count identities; a deployment with
        untrusted unauthenticated traffic should raise
        ``config.rate_limit.max_identities`` or terminate limits at a shared
        gateway/edge layer (this limiter is per-worker, not distributed).
        """
        if len(self._buckets) < self._max_identities:
            return
        expired = [
            key
            for key, (window_start, _count) in self._buckets.items()
            if now - window_start >= self._window_seconds
        ]
        for key in expired:
            del self._buckets[key]
        while len(self._buckets) >= self._max_identities:
            victim, _bucket = min(
                self._buckets.items(),
                key=lambda item: (item[1][1], item[1][0]),
            )
            del self._buckets[victim]

    def reset(self) -> None:
        """Clear all rate-limit state.  Useful in tests."""
        with self._lock:
            self._buckets.clear()

    @staticmethod
    def fingerprint(api_key: str) -> str:
        """Produce a safe, non-secret fingerprint from an API key.

        Uses SHA-256 truncated to 16 hex chars.  This is used as the
        rate-limit identity so the raw key is never stored or logged.
        """
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
