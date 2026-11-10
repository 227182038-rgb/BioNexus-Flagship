"""Tests for centralized error handling and observability in BioNexus."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from nexus.api.app import app, set_nexus
from nexus.core.exceptions import (
    BioNexusError,
    KnowledgeError,
    PubMedError,
    UniProtError,
    PDBError,
    EnsemblError,
    GOError,
    NCBIError,
    ValidationError as BioNexusValidationError,
    ConfigurationError,
)
from nexus.providers.base import ProviderError, ProviderTimeoutError
from nexus.sdk import Nexus


class TestExceptionHierarchy:
    """Verify specific knowledge exceptions inherit from KnowledgeError and BioNexusError."""

    def test_knowledge_exceptions_subclass_relationships(self) -> None:
        for exc_cls in (PubMedError, UniProtError, PDBError, EnsemblError, GOError, NCBIError):
            assert issubclass(exc_cls, KnowledgeError)
            assert issubclass(exc_cls, BioNexusError)
            assert issubclass(exc_cls, Exception)

    def test_provider_exceptions_subclass_relationships(self) -> None:
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(ProviderError, Exception)


class TestAPIErrorHandling:
    """Verify API centralized exception handlers, status codes, and correlation IDs."""

    @pytest.fixture
    def mock_nexus(self) -> Nexus:
        nx = MagicMock(spec=Nexus)
        return nx

    @pytest.fixture
    def client(self, mock_nexus: Nexus) -> TestClient:
        set_nexus(mock_nexus)
        return TestClient(app, raise_server_exceptions=False)

    def test_provider_error_returns_502_with_correlation_id(
        self, client: TestClient, mock_nexus: MagicMock
    ) -> None:
        mock_nexus.interpret.side_effect = ProviderError("Secret API key xyz failed upstream")

        response = client.post("/api/interpret", json={"question": "test?"})
        assert response.status_code == 502
        data = response.json()
        assert "correlation_id" in data
        assert len(data["correlation_id"]) > 0
        assert data["error"] == "Upstream provider request failed."
        # Verify secret/internal detail was NOT leaked
        assert "Secret API key xyz" not in response.text

    def test_knowledge_error_returns_502_with_correlation_id(
        self, client: TestClient, mock_nexus: MagicMock
    ) -> None:
        mock_nexus.interpret.side_effect = PubMedError("Internal PubMed query failed with db token secret123")

        response = client.post("/api/interpret", json={"question": "test?"})
        assert response.status_code == 502
        data = response.json()
        assert "correlation_id" in data
        assert data["error"] == "Knowledge source request failed."
        assert "secret123" not in response.text

    def test_validation_error_returns_400_with_correlation_id(
        self, client: TestClient, mock_nexus: MagicMock
    ) -> None:
        mock_nexus.interpret.side_effect = BioNexusValidationError("Invalid query format")

        response = client.post("/api/interpret", json={"question": "test?"})
        assert response.status_code == 400
        data = response.json()
        assert "correlation_id" in data
        assert data["error"] == "Input validation failed."

    def test_configuration_error_returns_500_with_correlation_id(
        self, client: TestClient, mock_nexus: MagicMock
    ) -> None:
        mock_nexus.interpret.side_effect = ConfigurationError("Missing LM Studio endpoint host configuration")

        response = client.post("/api/interpret", json={"question": "test?"})
        assert response.status_code == 500
        data = response.json()
        assert "correlation_id" in data
        assert data["error"] == "Internal configuration error."
        assert "LM Studio endpoint host" not in response.text

    def test_biokit_unknown_program_404(self, client: TestClient, mock_nexus: MagicMock) -> None:
        mock_nexus.run_biokit.side_effect = KeyError("nonexistent_prog")

        response = client.post("/api/biokit", json={"program": "nonexistent_prog", "inputs": {}})
        assert response.status_code == 404
        assert "BioKit program not found: nonexistent_prog" in response.json()["detail"]
