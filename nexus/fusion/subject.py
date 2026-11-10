"""Canonical query-subject selection for fused biomedical entities."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.fusion.models import BiomedicalEntity
from nexus.knowledge.router import QueryIntent, QueryRouter


@dataclass(frozen=True, slots=True)
class SubjectSelection:
    """Selected biological subject and its selection score."""

    entity: BiomedicalEntity
    score: float


class SubjectResolver:
    """Select the entity that best represents the user's query."""

    _TYPE_PRIORITY: dict[QueryIntent, tuple[str, ...]] = {
        QueryIntent.GENE_SYMBOL: (
            "gene",
            "protein",
            "publication",
            "structure",
            "ontology",
            "unknown",
        ),
        QueryIntent.UNIPROT_ID: (
            "protein",
            "gene",
            "publication",
            "structure",
            "ontology",
            "unknown",
        ),
        QueryIntent.ENSEMBL_ID: (
            "gene",
            "protein",
            "publication",
            "structure",
            "ontology",
            "unknown",
        ),
        QueryIntent.PUBMED_ID: (
            "publication",
            "gene",
            "protein",
            "unknown",
        ),
        QueryIntent.PDB_ID: (
            "structure",
            "protein",
            "gene",
            "publication",
            "unknown",
        ),
        QueryIntent.GO_ID: (
            "ontology",
            "gene",
            "protein",
            "publication",
            "unknown",
        ),
        QueryIntent.NCBI_ID: (
            "unknown",
            "gene",
            "protein",
            "publication",
        ),
        QueryIntent.FREE_TEXT: (
            "gene",
            "protein",
            "disease",
            "pathway",
            "compound",
            "drug",
            "publication",
            "structure",
            "ontology",
            "unknown",
        ),
    }

    def __init__(
        self,
        router: QueryRouter | None = None,
    ) -> None:
        self._router = router or QueryRouter()

    def select(
        self,
        query: str,
        entities: list[BiomedicalEntity],
    ) -> SubjectSelection | None:
        """Select the best biological subject for a query."""
        if not entities:
            return None

        route = self._router.route(query)
        priorities = self._TYPE_PRIORITY[route.intent]

        scored: list[tuple[float, BiomedicalEntity]] = []

        for entity in entities:
            score = self._score_entity(
                route.normalized,
                entity,
                priorities,
            )

            scored.append(
                (score, entity)
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].confidence,
            ),
            reverse=True,
        )

        score, entity = scored[0]

        return SubjectSelection(
            entity=entity,
            score=score,
        )

    def _score_entity(
        self,
        query: str,
        entity: BiomedicalEntity,
        priorities: tuple[str, ...],
    ) -> float:
        entity_type = self._entity_type(entity)

        try:
            type_rank = priorities.index(entity_type)
        except ValueError:
            type_rank = len(priorities)

        score = float(
            len(priorities) - type_rank
        ) * 10.0

        query_key = query.strip().casefold()

        candidates = [
            entity.name,
            entity.id,
            entity.canonical_id or "",
            entity.canonical_source or "",
            *entity.aliases,
            *entity.genes,
            *entity.proteins,
        ]

        for candidate in candidates:
            if not candidate:
                continue

            candidate_key = str(candidate).casefold()

            if candidate_key == query_key:
                score += 100.0

            elif query_key in candidate_key:
                score += 20.0

        score += entity.confidence

        if entity.canonical_id:
            score += 2.0

        return score

    @staticmethod
    def _entity_type(
        entity: BiomedicalEntity,
    ) -> str:
        value = entity.entity_type

        if hasattr(value, "value"):
            return str(value.value).casefold()

        return str(value).casefold()


__all__ = [
    "SubjectResolver",
    "SubjectSelection",
]
