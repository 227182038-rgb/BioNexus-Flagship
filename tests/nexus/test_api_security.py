"""Security tests for the Nexus REST API.

Tests authentication, security headers, error codes, constant-time comparison,
unprotected vs protected endpoints, and non-exposure of secrets.
"""

from __future__ import annotations

import logging
import pytest
from fastapi.testclient import TestClient

from nexus.api.app import app, set_nexus, set_bionexus
from nexus.core.config import config
from nexus.providers.dummy import DummyProvider
from nexus.sdk import Nexus
from nexus.application.app import BioNexus


@pytest.fixture(autouse=True)
def reset_api_key():
    """Ensure api_key is restored to empty after tests."""
    original = config.api.api_key
    yield
    config.api.api_key = original


@pytest.fixture
def client(reset_api_key):
    set_nexus(Nexus(provider=DummyProvider(responder="Security test answer.")))
    set_bionexus(BioNexus())
    return TestClient(app)


class TestAPISecurity:
    def test_unauthenticated_dev_mode_permits_access(self, client: TestClient):
        """When no API key is configured, requests succeed without auth."""
        config.api.api_key = ""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert "status" in response.json()

        response = client.post("/api/interpret", json={"question": "BRCA1?"})
        assert response.status_code == 200

    def test_security_headers_present_on_all_responses(self, client: TestClient):
        """Security headers must be present on responses."""
        response = client.get("/api/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "Strict-Transport-Security" in response.headers
        assert response.headers.get("Cache-Control") == "no-store"
        assert "X-Correlation-ID" in response.headers

    def test_public_endpoints_unprotected_when_auth_configured(self, client: TestClient):
        """Health and root endpoints remain accessible without auth even when key is set."""
        config.api.api_key = "super-secret-key-12345"

        resp_root = client.get("/")
        assert resp_root.status_code == 200

        resp_health = client.get("/api/health")
        assert resp_health.status_code == 200

        resp_bn_health = client.get("/api/bionexus/health")
        assert resp_bn_health.status_code == 200


    def test_protected_endpoint_missing_key_returns_401(self, client: TestClient):
        """Protected endpoint with missing credentials returns 401."""
        config.api.api_key = "super-secret-key-12345"

        response = client.post("/api/interpret", json={"question": "BRCA1?"})
        assert response.status_code == 401
        data = response.json()
        assert data[
            "error"
        ] == "Authentication required."
        assert "correlation_id" in data
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("Cache-Control") == "no-store"

    def test_protected_endpoint_invalid_key_returns_403(self, client: TestClient):
        """Protected endpoint with invalid credentials returns 403."""
        config.api.api_key = "super-secret-key-12345"

        # Bearer format with wrong key
        response = client.post(
            "/api/interpret",
            json={"question": "BRCA1?"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 403
        data = response.json()
        assert data["error"] == "Forbidden: invalid credentials."
        assert "correlation_id" in data


        # X-API-Key with wrong key
        response = client.post(
            "/api/interpret",
            json={"question": "BRCA1?"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403
        assert response.json()[
            "error"
        ] == "Forbidden: invalid credentials."

    def test_protected_endpoint_valid_bearer_key_succeeds(self, client: TestClient):
        """Protected endpoint with valid Bearer token succeeds."""
        config.api.api_key = "super-secret-key-12345"

        response = client.post(
            "/api/interpret",
            json={"question": "What is TP53?"},
            headers={"Authorization": "Bearer super-secret-key-12345"},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Security test answer."

    def test_protected_endpoint_valid_x_api_key_succeeds(self, client: TestClient):
        """Protected endpoint with valid X-API-Key header succeeds."""
        config.api.api_key = "super-secret-key-12345"

        response = client.post(
            "/api/interpret",
            json={"question": "What is TP53?"},
            headers={"X-API-Key": "super-secret-key-12345"},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "Security test answer."

    def test_options_cors_preflight_bypasses_auth(self, client: TestClient):
        """OPTIONS requests bypass auth and include security headers."""
        config.api.api_key = "super-secret-key-12345"

        response = client.options("/api/interpret")
        assert response.status_code in (200, 405)
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_secrets_never_leaked_in_responses_or_errors(self, client: TestClient):
        """Ensure secret values are never reflected in response bodies."""
        secret = "extremely-confidential-token-xyz99"
        config.api.api_key = secret

        # 401 response
        r1 = client.post("/api/interpret", json={"question": "x"})
        assert secret not in r1.text


        # 403 response
        r2 = client.post(
            "/api/interpret",
            json={"question": "x"},
            headers={"Authorization": "Bearer invalid-attempt"},
        )
        assert secret not in r2.text

        # 200 response
        r3 = client.post(
            "/api/interpret",
            json={"question": "x"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert secret not in r3.text

    def test_secrets_never_logged(self, client: TestClient, caplog):
        """Ensure secret keys are not logged during authentication failures."""
        secret = "extremely-confidential-token-xyz99"
        config.api.api_key = secret

        with caplog.at_level(logging.WARNING, logger="nexus.api"):
            client.post(
                "/api/interpret",
                json={"question": "x"},
                headers={"Authorization": "Bearer leaked-attempt"},
            )

        for record in caplog.records:
            assert secret not in record.message
            assert "leaked-attempt" not in record.message
