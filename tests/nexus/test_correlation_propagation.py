"""Tests for correlation ID propagation across all layers of the scientific analysis pipeline."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from nexus.api.app import app, set_bionexus
from nexus.application.app import BioNexus
from nexus.core.logging import (
    get_correlation_id,
    set_correlation_id,
    get_logger,
    CorrelationIdFilter,
)
from nexus.fusion.engine import FusionEngine
from nexus.graph.builder import GraphBuilder
from nexus.normalization.resolver import NormalizationResolver
from nexus.pipeline.context import ScientificAnalysisContext
from nexus.reasoning.engine import ReasoningEngine
from nexus.workflows.analyze import AnalyzeBiomedicalEntityWorkflow


@pytest.fixture(autouse=True)
def reset_correlation_id_context():
    """Reset the contextvar correlation ID before each test for test isolation."""
    set_correlation_id("")
    yield
    set_correlation_id("")


class TestCorrelationIdUtilities:
    """Test correlation ID getter, setter, and logging filter."""

    def test_get_correlation_id_generates_default(self) -> None:
        cid = get_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 12

    def test_set_and_get_correlation_id(self) -> None:
        set_correlation_id("test-cid-123")
        assert get_correlation_id() == "test-cid-123"

    def test_correlation_id_filter_attaches_to_record(self) -> None:
        logger = get_logger("test.filter")
        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord("test.filter", logging.INFO, "path", 1, "test message", (), None)

        set_correlation_id("filter-cid-abc")
        filter_obj.filter(record)
        assert getattr(record, "correlation_id", None) == "filter-cid-abc"


class TestWorkflowLayerCorrelationPropagation:
    """Test propagation from Workflow down to Fusion, Normalization, Graph, and Reasoning."""

    @pytest.mark.asyncio
    async def test_workflow_propagates_correlation_id_to_all_subsystems(self) -> None:
        test_cid = "corr-wf-test-42"
        workflow = AnalyzeBiomedicalEntityWorkflow()

        reasoning_calls: list[dict] = []
        orig_analyze = ReasoningEngine.analyze

        def spy_analyze(self, *args, **kwargs):
            reasoning_calls.append(kwargs)
            return orig_analyze(self, *args, **kwargs)

        with patch.object(workflow.fusion, "search", wraps=workflow.fusion.search) as spy_fusion_search, \
             patch.object(workflow.builder, "build", wraps=workflow.builder.build) as spy_graph_build, \
             patch.object(ReasoningEngine, "analyze", spy_analyze):

            ctx = await workflow.run_context("TP53", correlation_id=test_cid)

            # Check context and report metadata
            assert ctx.correlation_id == test_cid
            assert ctx.report is not None
            assert ctx.report.metadata.get("correlation_id") == test_cid

            # Verify fusion was called with correlation_id
            assert spy_fusion_search.call_count >= 1
            _, fusion_kwargs = spy_fusion_search.call_args
            assert fusion_kwargs.get("correlation_id") == test_cid

            # Verify graph builder was called with correlation_id
            assert spy_graph_build.call_count >= 1
            _, graph_kwargs = spy_graph_build.call_args
            assert graph_kwargs.get("correlation_id") == test_cid

            # Verify reasoning engine was called with correlation_id
            assert len(reasoning_calls) >= 1
            assert reasoning_calls[0].get("correlation_id") == test_cid

    @pytest.mark.asyncio
    async def test_workflow_generates_safe_local_id_when_none_provided(self) -> None:
        workflow = AnalyzeBiomedicalEntityWorkflow()
        ctx = await workflow.run_context("TP53", correlation_id=None)

        assert ctx.correlation_id != ""
        assert isinstance(ctx.correlation_id, str)
        assert len(ctx.correlation_id) == 12
        assert ctx.report is not None
        assert ctx.report.metadata.get("correlation_id") == ctx.correlation_id

    @pytest.mark.asyncio
    async def test_fusion_propagates_to_normalization(self) -> None:
        test_cid = "corr-fusion-norm-77"
        engine = FusionEngine([])

        with patch.object(engine._normalization, "resolve", wraps=engine._normalization.resolve) as spy_norm_resolve:
            await engine.search("TP53", limit=2, correlation_id=test_cid)

            assert spy_norm_resolve.call_count >= 1
            _, norm_kwargs = spy_norm_resolve.call_args
            assert norm_kwargs.get("correlation_id") == test_cid

    def test_graph_builder_propagates_to_literature(self) -> None:
        test_cid = "corr-graph-lit-88"
        builder = GraphBuilder()
        from nexus.fusion.models import BiomedicalEntity

        subject = BiomedicalEntity(id="TP53", name="TP53", entity_type="Gene")
        pub = BiomedicalEntity(id="pubmed:12345", name="Study", entity_type="Publication")

        with patch.object(builder.literature, "link", wraps=builder.literature.link) as spy_lit_link:
            builder.build([subject, pub], subject=subject, correlation_id=test_cid)

            assert spy_lit_link.call_count >= 1
            _, lit_kwargs = spy_lit_link.call_args
            assert lit_kwargs.get("correlation_id") == test_cid


class TestApplicationLayerPropagation:
    """Test BioNexus application entrypoints propagate correlation IDs."""

    @pytest.mark.asyncio
    async def test_bionexus_analyze_context_propagates_cid(self) -> None:
        app_instance = BioNexus()
        test_cid = "app-ctx-corr-11"

        with patch.object(app_instance.pipeline, "analyze_context", wraps=app_instance.pipeline.analyze_context) as spy_pipe:
            ctx = await app_instance.analyze_context("TP53", correlation_id=test_cid)
            assert ctx.correlation_id == test_cid
            assert spy_pipe.call_count == 1
            _, pipe_kwargs = spy_pipe.call_args
            assert pipe_kwargs.get("correlation_id") == test_cid

    def test_bionexus_analyze_sync_propagates_cid(self) -> None:
        app_instance = BioNexus()
        test_cid = "app-sync-corr-22"

        with patch.object(app_instance.pipeline, "run", wraps=app_instance.pipeline.run) as spy_run:
            report = app_instance.analyze("TP53", correlation_id=test_cid)
            assert report is not None
            assert spy_run.call_count == 1
            _, run_kwargs = spy_run.call_args
            assert run_kwargs.get("correlation_id") == test_cid


class TestAPILayerCorrelationPropagation:
    """Test HTTP API layer correlation headers, endpoints, and error responses."""

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app, raise_server_exceptions=False)

    def test_bionexus_analyze_propagates_client_correlation_id(self, client: TestClient) -> None:
        client_cid = "client-req-cid-9999"
        # Reset to default BioNexus to avoid any lingering mock
        set_bionexus(BioNexus())
        response = client.post(
            "/api/bionexus/analyze",
            json={"query": "TP53"},
            headers={"X-Correlation-ID": client_cid},
        )

        assert response.status_code == 200
        assert response.headers.get("X-Correlation-ID") == client_cid
        data = response.json()
        assert data["metadata"]["correlation_id"] == client_cid
        analysis_id = data["analysis_id"]
        assert analysis_id is not None
        assert analysis_id.startswith("ana_")

        # Query sub-resources and verify correlation headers
        graph_resp = client.get(
            f"/api/bionexus/graph/{analysis_id}",
            headers={"X-Correlation-ID": client_cid},
        )
        assert graph_resp.status_code == 200
        assert graph_resp.headers.get("X-Correlation-ID") == client_cid
        assert graph_resp.json()["analysis_id"] == analysis_id

        evidence_resp = client.get(
            f"/api/bionexus/evidence/{analysis_id}",
            headers={"X-Correlation-ID": client_cid},
        )
        assert evidence_resp.status_code == 200
        assert evidence_resp.headers.get("X-Correlation-ID") == client_cid

        lit_resp = client.get(
            f"/api/bionexus/literature/{analysis_id}",
            headers={"X-Correlation-ID": client_cid},
        )
        assert lit_resp.status_code == 200
        assert lit_resp.headers.get("X-Correlation-ID") == client_cid

        struct_resp = client.get(
            f"/api/bionexus/structures/{analysis_id}",
            headers={"X-Correlation-ID": client_cid},
        )
        assert struct_resp.status_code == 200
        assert struct_resp.headers.get("X-Correlation-ID") == client_cid

    def test_bionexus_analyze_generates_id_when_missing(self, client: TestClient) -> None:
        set_bionexus(BioNexus())
        response = client.post(
            "/api/bionexus/analyze",
            json={"query": "TP53"},
        )
        assert response.status_code == 200
        cid = response.headers.get("X-Correlation-ID")
        assert cid is not None
        assert len(cid) == 12
        assert response.json()["metadata"]["correlation_id"] == cid

    def test_no_secret_leakage_in_error_responses(self, client: TestClient) -> None:
        mock_bn = MagicMock(spec=BioNexus)
        mock_bn.analyze_context = AsyncMock(
            side_effect=RuntimeError("Connection refused with token bearer_token_secret_xyz123")
        )
        set_bionexus(mock_bn)

        client_cid = "leak-test-cid-000"
        response = client.post(
            "/api/bionexus/analyze",
            json={"query": "TP53"},
            headers={"X-Correlation-ID": client_cid},
        )

        assert response.status_code == 500
        assert response.headers.get("X-Correlation-ID") == client_cid
        assert "bearer_token_secret_xyz123" not in response.text
        assert response.json()["detail"] == "Internal server error"
