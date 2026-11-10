"""Query-aware entity ranking."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.graph.algorithms import GraphAlgorithms
from nexus.graph.graph import KnowledgeGraph


@dataclass(slots=True)
class RankedEntity:
    """Represents a graph entity with a relevance score."""

    entity_id: str
    score: float


class EntityRanker:
    """Ranks graph entities against a biomedical query and subject."""

    def __init__(
        self,
        graph: KnowledgeGraph,
    ) -> None:
        self.graph = graph
        self.algorithms = GraphAlgorithms(graph)

    def degree_score(
        self,
        entity_id: str,
    ) -> float:
        return float(
            self.algorithms.degree(entity_id)
        )

    def centrality_score(
        self,
        entity_id: str,
    ) -> float:
        return self.algorithms.degree_centrality().get(
            entity_id,
            0.0,
        )

    def relevance_score(
        self,
        entity_id: str,
        *,
        query: str = "",
        subject_id: str | None = None,
    ) -> float:
        """Calculate query-aware graph relevance."""

        node = self.graph.get_node(entity_id)

        if node is None:
            return 0.0

        query_key = query.strip().casefold()

        score = 0.0

        # 1. Direct subject match is the strongest relevance signal.
        if subject_id and entity_id == subject_id:
            score += 10.0

        # 2. Canonical identifier matching.
        canonical_id = str(
            node.properties.get(
                "canonical_id",
                "",
            )
        ).casefold()

        original_id = str(
            node.properties.get(
                "original_id",
                "",
            )
        ).casefold()

        if query_key:
            if query_key == entity_id.casefold():
                score += 8.0

            if query_key == canonical_id:
                score += 8.0

            if query_key == original_id:
                score += 6.0

            if query_key == node.label.casefold():
                score += 7.0

            for alias in node.aliases:
                alias_key = alias.casefold()

                if query_key == alias_key:
                    score += 6.0
                elif query_key in alias_key:
                    score += 2.0

        # 3. Canonical entities receive a modest identity bonus.
        if canonical_id:
            score += 1.0

        # 4. Graph structure remains useful, but cannot dominate.
        degree = self.degree_score(entity_id)
        centrality = self.centrality_score(entity_id)

        score += min(degree, 10.0) * 0.20
        score += centrality * 2.0

        # 5. Explicit node confidence contributes modestly.
        score += max(
            0.0,
            min(
                float(node.confidence),
                1.0,
            ),
        )

        # 6. Evidence coverage.
        score += min(
            len(node.evidence),
            10,
        ) * 0.10

        return score

    def rank(
        self,
        *,
        query: str = "",
        subject_id: str | None = None,
    ) -> list[RankedEntity]:
        """Rank graph entities for a specific scientific query."""

        results: list[RankedEntity] = []

        for node in self.graph.nodes.values():
            score = self.relevance_score(
                node.id,
                query=query,
                subject_id=subject_id,
            )

            results.append(
                RankedEntity(
                    entity_id=node.id,
                    score=score,
                )
            )

        return sorted(
            results,
            key=lambda item: (
                item.score,
                item.entity_id,
            ),
            reverse=True,
        )


__all__ = [
    "EntityRanker",
    "RankedEntity",
]
