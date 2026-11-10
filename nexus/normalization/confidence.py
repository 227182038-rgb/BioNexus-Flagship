"""Confidence scoring for canonical biomedical entities.

Calculates transparent, bounded confidence scores from evidence and identity
signals already present in a BiomedicalEntity.

This module does not claim biological truth. It measures confidence in the
quality and consistency of the currently available representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.fusion.models import BiomedicalEntity


@dataclass(frozen=True, slots=True)
class ConfidenceBreakdown:
    """Explainable confidence components."""

    evidence: float
    identity: float
    cross_references: float
    biological_context: float
    overall: float


class ConfidenceEngine:
    """Calculate explainable confidence for biomedical entities."""

    def __init__(
        self,
        *,
        evidence_weight: float = 0.40,
        identity_weight: float = 0.30,
        cross_reference_weight: float = 0.20,
        context_weight: float = 0.10,
    ) -> None:
        weights = (
            evidence_weight,
            identity_weight,
            cross_reference_weight,
            context_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError("confidence weights cannot be negative")

        total = sum(weights)

        if total <= 0:
            raise ValueError("confidence weights must have a positive total")

        self._weights = tuple(
            weight / total
            for weight in weights
        )

    def score(
        self,
        entity: BiomedicalEntity,
    ) -> float:
        """Return a confidence score in the inclusive range [0, 1]."""
        return self.breakdown(entity).overall

    def breakdown(
        self,
        entity: BiomedicalEntity,
    ) -> ConfidenceBreakdown:
        """Return the explainable confidence components."""
        if not isinstance(entity, BiomedicalEntity):
            raise TypeError("entity must be a BiomedicalEntity")

        evidence = self._evidence_score(entity)
        identity = self._identity_score(entity)
        cross_references = self._cross_reference_score(entity)
        context = self._context_score(entity)

        evidence_weight, identity_weight, xref_weight, context_weight = (
            self._weights
        )

        overall = (
            evidence * evidence_weight
            + identity * identity_weight
            + cross_references * xref_weight
            + context * context_weight
        )

        return ConfidenceBreakdown(
            evidence=self._clamp(evidence),
            identity=self._clamp(identity),
            cross_references=self._clamp(cross_references),
            biological_context=self._clamp(context),
            overall=self._clamp(overall),
        )

    def apply(
        self,
        entity: BiomedicalEntity,
    ) -> BiomedicalEntity:
        """Calculate and store the confidence on an entity."""
        entity.confidence = self.score(entity)
        return entity

    def apply_many(
        self,
        entities: list[BiomedicalEntity],
    ) -> list[BiomedicalEntity]:
        """Calculate confidence for multiple entities."""
        for entity in entities:
            self.apply(entity)

        return entities

    @staticmethod
    def _evidence_score(
        entity: BiomedicalEntity,
    ) -> float:
        if not entity.evidence:
            return 0.0

        confidences = [
            ConfidenceEngine._clamp(evidence.confidence)
            for evidence in entity.evidence
        ]

        # Additional independent evidence increases confidence, but with
        # diminishing returns rather than allowing the score to exceed 1.
        mean_confidence = sum(confidences) / len(confidences)
        coverage = 1.0 - (1.0 / (len(confidences) + 1.0))

        return ConfidenceEngine._clamp(
            mean_confidence * (0.5 + 0.5 * coverage)
        )

    @staticmethod
    def _identity_score(
        entity: BiomedicalEntity,
    ) -> float:
        score = 0.0

        if entity.id:
            score += 0.30

        if entity.canonical_id:
            score += 0.50

        if entity.canonical_source:
            score += 0.20

        return ConfidenceEngine._clamp(score)

    @staticmethod
    def _cross_reference_score(
        entity: BiomedicalEntity,
    ) -> float:
        count = len(entity.cross_references)

        if count == 0:
            return 0.0

        # Multiple independent cross-references increase confidence with
        # diminishing returns.
        return ConfidenceEngine._clamp(
            1.0 - (1.0 / (count + 1.0))
        )

    @staticmethod
    def _context_score(
        entity: BiomedicalEntity,
    ) -> float:
        signals = 0

        if entity.name:
            signals += 1

        if entity.description:
            signals += 1

        if entity.aliases:
            signals += 1

        if entity.genes or entity.proteins:
            signals += 1

        if entity.structures or entity.pathways:
            signals += 1

        if entity.diseases or entity.publications:
            signals += 1

        if entity.ontology_terms:
            signals += 1

        if entity.sequences:
            signals += 1

        return ConfidenceEngine._clamp(
            signals / 8.0
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


__all__ = [
    "ConfidenceBreakdown",
    "ConfidenceEngine",
]
