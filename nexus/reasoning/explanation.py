"""Scientific explanation engine."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceManager
from .summarizer import ScientificSummary


@dataclass(slots=True)
class ScientificExplanation:
    """Human-readable explanation backed by evidence."""

    summary: str

    reasoning: list[str]

    evidence: list[str]

    confidence: float


class ExplanationEngine:
    """Generate explainable scientific reports."""

    def explain(
        self,
        summary: ScientificSummary,
        evidence_manager: EvidenceManager,
    ) -> ScientificExplanation:

        reasoning: list[str] = []

        if summary.overview:
            reasoning.append(
                summary.overview
            )

        reasoning.extend(
            summary.key_points
        )

        evidence: list[str] = []

        for item in evidence_manager.top(10):
            evidence.append(
                f"{item.source}: "
                f"{item.identifier} — "
                f"{item.title} "
                f"(confidence={item.confidence:.2f})"
            )

        return ScientificExplanation(
            summary=summary.title,
            reasoning=reasoning,
            evidence=evidence,
            confidence=summary.confidence,
        )


__all__ = [
    "ExplanationEngine",
    "ScientificExplanation",
]
