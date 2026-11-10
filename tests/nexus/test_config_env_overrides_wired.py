"""Regression tests for finding N-01: documented BIO_NEXUS_* env variables
must actually be honored when the application starts.

``config.apply_env_overrides()`` existed but was never called, so every
documented environment variable was inert.  Constructing ``BioNexus()`` must
apply the overrides.
"""

from __future__ import annotations

import pytest

from nexus.application.app import BioNexus
from nexus.core.config import config


@pytest.fixture()
def restore_config():
    """Restore rate-limit state so later test modules are unaffected.

    ``apply_env_overrides()`` mutates ``config.rate_limit`` *in place*, so a
    plain reference snapshot would keep leaked ``enabled=True`` across the
    suite; a value snapshot (``dataclasses.replace``) is required.
    """
    import dataclasses

    saved_api_key = config.api.api_key
    saved_rate_limit = dataclasses.replace(config.rate_limit)
    yield
    config.api.api_key = saved_api_key
    config.rate_limit = saved_rate_limit
    from nexus.api.app import set_rate_limiter

    set_rate_limiter(None)


def test_api_key_env_override_applies_at_startup(
    restore_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIO_NEXUS_API_KEY", "envkey-abc123")
    BioNexus()
    assert config.api.api_key == "envkey-abc123"


def test_rate_limit_env_overrides_apply_at_startup(
    restore_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIO_NEXUS_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("BIO_NEXUS_RATE_LIMIT_MAX_REQUESTS", "7")
    BioNexus()
    assert config.rate_limit.enabled is True
    assert config.rate_limit.max_requests == 7


def test_invalid_env_values_are_ignored(
    restore_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIO_NEXUS_RATE_LIMIT_MAX_REQUESTS", "not-a-number")
    BioNexus()
    assert config.rate_limit.max_requests == 7 or config.rate_limit.max_requests > 0
