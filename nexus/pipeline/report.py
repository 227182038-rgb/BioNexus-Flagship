"""Scientific analysis report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from nexus.reasoning.confidence import ScientificConfidence
from nexus.reasoning.contradiction import Contradiction
from nexus.reasoning.explanation import ScientificExplanation
from nexus.reasoning.hypothesis import ResearchHypothesis
from nexus.reasoning.ranking import RankedEntity


@dataclass(slots=True)
class ScientificReport:
    """Complete BioNexus scientific analysis report."""

    query: str

    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    ranked_entities: list[RankedEntity] = field(
        default_factory=list
    )

    explanation: ScientificExplanation | None = None

    contradictions: list[Contradiction] = field(
        default_factory=list
    )

    hypotheses: list[ResearchHypothesis] = field(
        default_factory=list
    )

    scientific_confidence: ScientificConfidence | None = None

    metadata: dict[str, str] = field(
        default_factory=dict
    )


__all__ = [
    "ScientificReport",
]
