"""Scientific reasoning engine."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.fusion.models import BiomedicalEntity
from nexus.graph.graph import KnowledgeGraph
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id

from .confidence import (
    ScientificConfidence,
    ScientificConfidenceEngine,
)
from .contradiction import (
    Contradiction,
    ContradictionDetector,
)
from .evidence import EvidenceManager
from .explanation import (
    ExplanationEngine,
    ScientificExplanation,
)
from .hypothesis import (
    HypothesisGenerator,
    ResearchHypothesis,
)
from .ranking import (
    EntityRanker,
    RankedEntity,
)
from .summarizer import ScientificSummarizer


@dataclass(slots=True)
class ReasoningReport:
    """Complete reasoning output."""

    ranked_entities: list[RankedEntity]

    explanation: ScientificExplanation

    contradictions: list[Contradiction]

    hypotheses: list[ResearchHypothesis]

    scientific_confidence: ScientificConfidence


class ReasoningEngine:
    """Coordinates all scientific reasoning modules."""

    def __init__(
        self,
        graph: KnowledgeGraph,
    ) -> None:
        self.logger = get_logger("reasoning")
        self.graph = graph

        self.ranker = EntityRanker(graph)
        self.summarizer = ScientificSummarizer()
        self.explainer = ExplanationEngine()
        self.contradictions = ContradictionDetector()
        self.hypotheses = HypothesisGenerator(graph)
        self.confidence = ScientificConfidenceEngine()

    def analyze(
        self,
        entity: BiomedicalEntity,
        evidence: EvidenceManager,
        *,
        query: str = "",
        correlation_id: str | None = None,
    ) -> ReasoningReport:
        """Analyze a canonical biomedical subject."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        subject_id = (
            entity.canonical_id
            or entity.id
        )

        self.logger.info("[%s] Reasoning analysis for subject %s (query=%r)", cid, subject_id, query)

        try:
            ranked = self.ranker.rank(
                query=query,
                subject_id=subject_id,
            )

            summary = self.summarizer.summarize(
                entity,
                evidence,
            )

            explanation = self.explainer.explain(
                summary,
                evidence,
            )

            contradiction_list = (
                self.contradictions.detect(
                    evidence.all(),
                )
            )

            hypotheses = (
                self.hypotheses.generate(
                    subject_id=subject_id,
                )
            )

            scientific_confidence = (
                self.confidence.calculate(
                    evidence.all(),
                )
            )

            return ReasoningReport(
                ranked_entities=ranked,
                explanation=explanation,
                contradictions=contradiction_list,
                hypotheses=hypotheses,
                scientific_confidence=scientific_confidence,
            )
        except Exception as exc:
            self.logger.error("Reasoning analysis error for %s [%s]: %s", subject_id, cid, exc)
            raise


__all__ = [
    "ReasoningEngine",
    "ReasoningReport",
]
