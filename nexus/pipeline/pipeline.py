"""BioNexus execution pipeline."""

from __future__ import annotations

import asyncio

from nexus.pipeline.context import ScientificAnalysisContext
from nexus.pipeline.report import ScientificReport
from nexus.workflows.analyze import AnalyzeBiomedicalEntityWorkflow


class BioNexusPipeline:
    """Central execution pipeline."""

    def __init__(self) -> None:
        self._analyze = AnalyzeBiomedicalEntityWorkflow()

    def analyze(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> ScientificReport:
        """Analyze a biomedical query and return the completed report.

        Runs the (async) workflow to completion on a fresh event loop.
        Must be called from synchronous contexts, or through
        :meth:`nexus.application.app.BioNexus.analyze` when an event loop is
        already running (which transparently defers to a worker thread).
        """
        return asyncio.run(self._analyze.run(query, correlation_id=correlation_id))

    async def analyze_context(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> ScientificAnalysisContext:
        """Return the complete reusable scientific analysis context."""
        return await self._analyze.run_context(query, correlation_id=correlation_id)

    def run(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> ScientificReport:
        """Alias of :meth:`analyze` (kept for backward compatibility)."""
        return self.analyze(query, correlation_id=correlation_id)


__all__ = [
    "BioNexusPipeline",
]
