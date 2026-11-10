"""Evidence-aware scientific summarization utilities."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.fusion.models import BiomedicalEntity

from .evidence import EvidenceManager


@dataclass(slots=True)
class ScientificSummary:
    """Structured scientific summary of a biomedical subject."""

    title: str

    overview: str

    key_points: list[str]

    evidence_count: int

    confidence: float


class ScientificSummarizer:
    """Create evidence-aware scientific summaries."""

    def summarize(
        self,
        entity: BiomedicalEntity,
        evidence: EvidenceManager,
    ) -> ScientificSummary:
        """Build a structured summary from entity data and evidence."""

        entity_type = self._entity_type(entity)

        overview_parts: list[str] = []

        if entity.name:
            overview_parts.append(
                f"{entity.name} is represented as a "
                f"{entity_type} in BioNexus."
            )

        if entity.description:
            description = entity.description.strip()

            if description:
                overview_parts.append(
                    description
                )

        points: list[str] = []

        if entity.canonical_id:
            points.append(
                f"Canonical identity: {entity.canonical_id}"
            )

        if entity.canonical_source:
            points.append(
                f"Canonical source: {entity.canonical_source}"
            )

        if entity.genes:
            points.append(
                "Associated genes: "
                + ", ".join(entity.genes)
            )

        if entity.proteins:
            points.append(
                "Associated proteins: "
                + ", ".join(entity.proteins)
            )

        if entity.diseases:
            points.append(
                "Associated diseases: "
                + ", ".join(entity.diseases)
            )

        if entity.pathways:
            points.append(
                "Biological pathways: "
                + ", ".join(entity.pathways)
            )

        if entity.structures:
            points.append(
                f"Available structures: "
                f"{len(entity.structures)}"
            )

        if entity.ontology_terms:
            points.append(
                "Ontology terms: "
                + ", ".join(entity.ontology_terms)
            )

        if entity.sequences:
            points.append(
                f"Sequence records: {len(entity.sequences)}"
            )

        if entity.cross_references:
            points.append(
                f"Cross-references: "
                f"{len(entity.cross_references)}"
            )

        evidence_items = evidence.all()

        if evidence_items:
            source_counts: dict[str, int] = {}

            for item in evidence_items:
                source_counts[item.source] = (
                    source_counts.get(item.source, 0)
                    + 1
                )

            source_summary = ", ".join(
                f"{source}: {count}"
                for source, count in sorted(
                    source_counts.items()
                )
            )

            points.append(
                f"Evidence sources: {source_summary}"
            )

            top_items = evidence.top(5)

            for item in top_items:
                points.append(
                    f"Evidence: {item.source} "
                    f"{item.identifier} — {item.title}"
                )

        confidence = max(
            0.0,
            min(
                1.0,
                float(entity.confidence),
            ),
        )

        return ScientificSummary(
            title=(
                entity.name
                or entity.canonical_id
                or entity.id
            ),
            overview=" ".join(
                overview_parts
            ),
            key_points=points,
            evidence_count=len(
                evidence_items
            ),
            confidence=confidence,
        )

    @staticmethod
    def _entity_type(
        entity: BiomedicalEntity,
    ) -> str:
        value = entity.entity_type

        if hasattr(value, "value"):
            return str(value.value).lower()

        return str(value).lower()


__all__ = [
    "ScientificSummary",
    "ScientificSummarizer",
]
