"""Regression tests for lead L-12: analyze must return a ScientificReport,
not an un-awaited coroutine.

The pipeline previously annotated ``analyze(...) -> ScientificReport`` while
returning the raw coroutine of the workflow; callers masked this by passing
the coroutine into ``asyncio.run``.  The contract must hold directly.
"""

from __future__ import annotations

import inspect

from nexus.pipeline.pipeline import BioNexusPipeline


def test_pipeline_analyze_returns_report_object() -> None:
    pipeline = BioNexusPipeline()
    report = pipeline.analyze("TP53")
    assert not inspect.iscoroutine(report)
    assert report is not None


def test_pipeline_run_returns_report_object() -> None:
    pipeline = BioNexusPipeline()
    report = pipeline.run("TP53")
    assert not inspect.iscoroutine(report)
    assert report is not None
