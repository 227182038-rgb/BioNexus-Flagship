"""Shared analysis context for BioNexus scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.fusion.models import BiomedicalEntity
from nexus.graph.graph import KnowledgeGraph
from nexus.pipeline.report import ScientificReport
from nexus.reasoning.evidence import Evidence


@dataclass(slots=True)
class ScientificAnalysisContext:
    """Reusable result of one complete BioNexus analysis."""

    query: str

    subject: BiomedicalEntity | None

    entities: list[BiomedicalEntity] = field(
        default_factory=list
    )

    graph: KnowledgeGraph | None = None

    evidence: list[Evidence] = field(
        default_factory=list
    )

    report: ScientificReport | None = None

    correlation_id: str = ""


__all__ = [
    "ScientificAnalysisContext",
]
