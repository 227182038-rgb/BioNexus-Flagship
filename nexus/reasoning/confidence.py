"""Scientific conclusion confidence calibration."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import Evidence


@dataclass(frozen=True, slots=True)
class ScientificConfidence:
    """Explainable confidence in a reasoning context."""

    evidence_quality: float
    source_diversity: float
    relationship_support: float
    literature_support: float
    consistency: float
    overall: float


class ScientificConfidenceEngine:
    """Calculate confidence in an evidence-backed conclusion."""

    def calculate(
        self,
        evidence: list[Evidence],
    ) -> ScientificConfidence:
        """Calculate bounded scientific confidence."""

        if not evidence:
            return ScientificConfidence(
                evidence_quality=0.0,
                source_diversity=0.0,
                relationship_support=0.0,
                literature_support=0.0,
                consistency=1.0,
                overall=0.0,
            )

        evidence_quality = self._evidence_quality(
            evidence
        )

        source_diversity = self._source_diversity(
            evidence
        )

        relationship_support = self._relationship_support(
            evidence
        )

        literature_support = self._literature_support(
            evidence
        )

        consistency = self._consistency(
            evidence
        )

        overall = (
            evidence_quality * 0.30
            + source_diversity * 0.20
            + relationship_support * 0.20
            + literature_support * 0.20
            + consistency * 0.10
        )

        return ScientificConfidence(
            evidence_quality=self._clamp(
                evidence_quality
            ),
            source_diversity=self._clamp(
                source_diversity
            ),
            relationship_support=self._clamp(
                relationship_support
            ),
            literature_support=self._clamp(
                literature_support
            ),
            consistency=self._clamp(
                consistency
            ),
            overall=self._clamp(
                overall
            ),
        )

    @staticmethod
    def _evidence_quality(
        evidence: list[Evidence],
    ) -> float:
        values = [
            ScientificConfidenceEngine._clamp(
                item.confidence
            )
            for item in evidence
        ]

        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    @staticmethod
    def _source_diversity(
        evidence: list[Evidence],
    ) -> float:
        sources = {
            item.source.casefold()
            for item in evidence
            if item.source
        }

        # Saturating diversity function.
        return min(
            1.0,
            len(sources) / 4.0,
        )

    @staticmethod
    def _relationship_support(
        evidence: list[Evidence],
    ) -> float:
        graph_items = [
            item
            for item in evidence
            if item.source.casefold() == "graph"
        ]

        if not graph_items:
            return 0.0

        values = [
            ScientificConfidenceEngine._clamp(
                item.confidence
            )
            for item in graph_items
        ]

        return min(
            1.0,
            (
                sum(values) / len(values)
            )
            + min(
                0.20,
                0.05 * max(
                    0,
                    len(values) - 1,
                ),
            ),
        )

    @staticmethod
    def _literature_support(
        evidence: list[Evidence],
    ) -> float:
        literature = [
            item
            for item in evidence
            if item.source.casefold() == "pubmed"
        ]

        if not literature:
            return 0.0

        values = [
            ScientificConfidenceEngine._clamp(
                item.confidence
            )
            for item in literature
        ]

        return min(
            1.0,
            (
                sum(values) / len(values)
            )
            * min(
                1.0,
                0.50
                + 0.10 * len(values),
            ),
        )

    @staticmethod
    def _consistency(
        evidence: list[Evidence],
    ) -> float:
        contradictions = [
            item
            for item in evidence
            if item.metadata.get(
                "contradiction",
                "",
            ).casefold() == "true"
        ]

        if not contradictions:
            return 1.0

        return max(
            0.0,
            1.0
            - min(
                1.0,
                0.20 * len(contradictions),
            ),
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )


__all__ = [
    "ScientificConfidence",
    "ScientificConfidenceEngine",
]
