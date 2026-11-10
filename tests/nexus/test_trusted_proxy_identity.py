"""Regression tests for lead L-04: proxy-aware rate-limit identity.

Defect: the API rate limiter keyed every client on the direct TCP peer
(``ip:<peer>``) only.  Behind a reverse proxy, ALL clients share one bucket
and a proxy-collapse outage follows.  A spoofable ``X-Forwarded-For`` fix
would be worse, so the design is:

- ``config.api.trusted_proxies`` (default ``[]``) lists peers whose
  ``X-Forwarded-For`` may be trusted;
- when the direct peer is NOT trusted (or no proxies are configured), the
  header is ignored entirely — spoofed ``X-Forwarded-For`` cannot change the
  rate-limit identity;
- when the direct peer IS trusted, the identity is the rightmost
  untrusted hop of ``X-Forwarded-For``; if every advertised hop is trusted,
  the leftmost hop is used.

TestClient presents peer host ``"testclient"``, which we use as "the proxy".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from nexus.api.app import app, set_rate_limiter
from nexus.core.config import config

URL = "/api/agents"


@pytest.fixture()
def limited(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enable rate limiting with a 1-request window and clean global state."""
    monkeypatch.setattr(config, "rate_limit", dataclasses.replace(config.rate_limit))
    monkeypatch.setattr(config, "api", dataclasses.replace(config.api))
    config.rate_limit.enabled = True
    config.rate_limit.max_requests = 1
    config.rate_limit.window_seconds = 60
    config.rate_limit.max_identities = 100
    set_rate_limiter(None)
    try:
        yield
    finally:
        set_rate_limiter(None)


def test_spoofed_xff_ignored_without_trusted_proxies(limited: None) -> None:
    """Without trusted proxies, XFF must NOT mint new identity buckets."""
    r1 = TestClient(app).get(URL, headers={"X-Forwarded-For": "1.2.3.4"})
    assert r1.status_code == 200
    r2 = TestClient(app).get(URL, headers={"X-Forwarded-For": "5.6.7.8"})
    assert r2.status_code == 429  # same shared bucket as r1


def test_trusted_proxy_yields_distinct_forwarded_identities(limited: None) -> None:
    """Behind a trusted proxy, distinct XFF clients get distinct buckets."""
    config.api.trusted_proxies = ["testclient"]
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get(URL, headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 200
    assert c.get(URL, headers={"X-Forwarded-For": "5.6.7.8"}).status_code == 200
    # 1.2.3.4 already consumed its single allowed request
    assert c.get(URL, headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 429


def test_rightmost_untrusted_hop_wins(limited: None) -> None:
    """With a trusted proxy chain, identity = rightmost untrusted hop."""
    config.api.trusted_proxies = ["testclient"]
    c = TestClient(app, raise_server_exceptions=False)
    chained = {"X-Forwarded-For": "1.2.3.4, 9.9.9.9"}
    assert c.get(URL, headers=chained).status_code == 200
    assert c.get(URL, headers=chained).status_code == 429  # 9.9.9.9 bucket full
    different = {"X-Forwarded-For": "8.8.8.8, 1.2.3.4"}
    assert c.get(URL, headers=different).status_code == 200  # hop 1.2.3.4


def test_all_trusted_hops_fall_back_to_leftmost(limited: None) -> None:
    """If every advertised hop is trusted, fall back to the leftmost hop."""
    config.api.trusted_proxies = ["testclient", "10.0.0.1"]
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get(URL, headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 200
    # No XFF at all -> direct peer (the proxy itself), a DIFFERENT bucket.
    assert c.get(URL).status_code == 200


def test_env_override_reads_trusted_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nexus.core.config as cfg_module

    snapshot = dataclasses.replace(cfg_module.config.api)
    try:
        monkeypatch.setenv("BIO_NEXUS_API_TRUSTED_PROXIES", "10.0.0.1, 10.0.0.2")
        cfg_module.config.apply_env_overrides()
        assert cfg_module.config.api.trusted_proxies == ["10.0.0.1", "10.0.0.2"]
    finally:
        cfg_module.config.api = snapshot
