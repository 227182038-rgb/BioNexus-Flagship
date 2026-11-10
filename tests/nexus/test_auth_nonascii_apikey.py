"""Regression tests for lead L-01: API-key handling must never yield a 500.

Any byte sequence a client may place in an API-key header (including obs-text
values 0x80-0xFF, which HTTP/1.1 permits) must produce 401/403, never an
unhandled exception.  Tests drive the ASGI app directly through TestClient
using raw byte header tuples, which is how the original 500 was reproduced.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexus.api.app import app
from nexus.core.config import config

PROTECTED = "/api/agents"
VALID_KEY = "testkey123"


@pytest.fixture()
def client() -> TestClient:
    tester = TestClient(app, raise_server_exceptions=False)
    previous = config.api.api_key
    config.api.api_key = VALID_KEY
    try:
        yield tester
    finally:
        config.api.api_key = previous


def test_nonascii_api_key_is_rejected_not_500(client: TestClient) -> None:
    response = client.get(PROTECTED, headers=[(b"x-api-key", b"se\xc3\xb1or")])
    assert response.status_code != 500
    assert response.status_code in (401, 403)


def test_nonascii_bearer_key_is_rejected_not_500(client: TestClient) -> None:
    response = client.get(PROTECTED, headers=[(b"authorization", b"Bearer se\xc3\xb1or")])
    assert response.status_code != 500
    assert response.status_code in (401, 403)


def test_ascii_wrong_key_yields_403(client: TestClient) -> None:
    assert client.get(PROTECTED, headers={"x-api-key": "wrongkey"}).status_code == 403


def test_ascii_valid_key_yields_200(client: TestClient) -> None:
    assert client.get(PROTECTED, headers={"x-api-key": VALID_KEY}).status_code == 200


def test_missing_key_yields_401(client: TestClient) -> None:
    assert client.get(PROTECTED).status_code == 401


def test_tab_in_key_is_rejected_not_500(client: TestClient) -> None:
    # HT is a legal obs-text byte; must not crash the comparison.
    response = client.get(PROTECTED, headers=[(b"x-api-key", b"a\tb")])
    assert response.status_code != 500
    assert response.status_code in (401, 403)
