"""Evidence-grounded research hypothesis generation."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.graph.edge import EdgeType
from nexus.graph.graph import KnowledgeGraph


@dataclass(slots=True)
class ResearchHypothesis:
    """Represents a generated research hypothesis."""

    title: str

    description: str

    supporting_entities: list[str]

    confidence: float

    rationale: str


class HypothesisGenerator:
    """Generate conservative hypotheses from graph relationships."""

    def __init__(
        self,
        graph: KnowledgeGraph,
    ) -> None:
        self.graph = graph

    def generate(
        self,
        subject_id: str | None = None,
    ) -> list[ResearchHypothesis]:
        """Generate hypotheses for a biological subject."""

        if subject_id is not None:
            return self._generate_for_subject(
                subject_id
            )

        return self._generate_from_connected_nodes()

    def _generate_for_subject(
        self,
        subject_id: str,
    ) -> list[ResearchHypothesis]:
        if not self.graph.has_node(subject_id):
            return []

        hypotheses: list[ResearchHypothesis] = []

        outgoing = self.graph.outgoing_edges(
            subject_id
        )

        for edge in outgoing:
            if edge.relation == EdgeType.ENCODES:
                hypothesis = self._encoding_hypothesis(
                    edge
                )

                if hypothesis is not None:
                    hypotheses.append(hypothesis)

            elif edge.relation == EdgeType.SUPPORTS:
                hypothesis = self._literature_hypothesis(
                    edge
                )

                if hypothesis is not None:
                    hypotheses.append(hypothesis)

            elif edge.relation in {
                EdgeType.INTERACTS_WITH,
                EdgeType.TARGETED_BY,
                EdgeType.PARTICIPATES_IN,
                EdgeType.ASSOCIATED_WITH,
            }:
                hypothesis = self._association_hypothesis(
                    edge
                )

                if hypothesis is not None:
                    hypotheses.append(hypothesis)

        return self._deduplicate(
            hypotheses
        )

    def _encoding_hypothesis(
        self,
        edge,
    ) -> ResearchHypothesis | None:
        source = self.graph.get_node(
            edge.source
        )

        target = self.graph.get_node(
            edge.target
        )

        if source is None or target is None:
            return None

        confidence = self._edge_confidence(
            edge
        )

        return ResearchHypothesis(
            title=(
                f"{source.label} may encode "
                f"{target.label}"
            ),
            description=(
                f"The graph contains an authoritative "
                f"ENCODES relationship from "
                f"{source.label} to {target.label}."
            ),
            supporting_entities=[
                source.id,
                target.id,
            ],
            confidence=confidence,
            rationale=(
                "The relationship is supported by "
                "authoritative cross-reference evidence."
            ),
        )

    def _literature_hypothesis(
        self,
        edge,
    ) -> ResearchHypothesis | None:
        source = self.graph.get_node(
            edge.source
        )

        target = self.graph.get_node(
            edge.target
        )

        if source is None or target is None:
            return None

        relevance = max(
            0.0,
            min(
                1.0,
                float(edge.weight),
            ),
        )

        if relevance < 0.50:
            return None

        return ResearchHypothesis(
            title=(
                f"{target.label} provides "
                f"literature support for {source.label}"
            ),
            description=(
                f"The publication is linked to "
                f"{source.label} through a literature "
                f"relevance assessment."
            ),
            supporting_entities=[
                source.id,
                target.id,
            ],
            confidence=(
                0.50
                + 0.50 * relevance
            ),
            rationale=(
                "The hypothesis is derived from an "
                "evidence-scored literature relationship, "
                "not from graph connectivity alone."
            ),
        )

    def _association_hypothesis(
        self,
        edge,
    ) -> ResearchHypothesis | None:
        source = self.graph.get_node(
            edge.source
        )

        target = self.graph.get_node(
            edge.target
        )

        if source is None or target is None:
            return None

        confidence = self._edge_confidence(
            edge
        )

        if confidence < 0.50:
            return None

        return ResearchHypothesis(
            title=(
                f"{source.label} may be biologically "
                f"associated with {target.label}"
            ),
            description=(
                f"The graph contains a "
                f"{edge.relation.value} relationship "
                f"between the two entities."
            ),
            supporting_entities=[
                source.id,
                target.id,
            ],
            confidence=confidence,
            rationale=(
                "This is a graph-derived research hypothesis "
                "and requires independent biological validation."
            ),
        )

    def _generate_from_connected_nodes(
        self,
    ) -> list[ResearchHypothesis]:
        hypotheses: list[ResearchHypothesis] = []

        for node in self.graph.nodes.values():
            degree = (
                len(
                    self.graph.outgoing_edges(
                        node.id
                    )
                )
                + len(
                    self.graph.incoming_edges(
                        node.id
                    )
                )
            )

            # Connectivity is now only a discovery signal.
            # It cannot directly become a hypothesis.
            if degree < 2:
                continue

            for edge in self.graph.outgoing_edges(
                node.id
            ):
                if edge.relation in {
                    EdgeType.ENCODES,
                    EdgeType.SUPPORTS,
                    EdgeType.INTERACTS_WITH,
                    EdgeType.TARGETED_BY,
                    EdgeType.PARTICIPATES_IN,
                    EdgeType.ASSOCIATED_WITH,
                }:
                    candidate = self._candidate_from_edge(
                        edge
                    )

                    if candidate is not None:
                        hypotheses.append(
                            candidate
                        )

        return self._deduplicate(
            hypotheses
        )

    def _candidate_from_edge(
        self,
        edge,
    ) -> ResearchHypothesis | None:
        if edge.relation == EdgeType.ENCODES:
            return self._encoding_hypothesis(
                edge
            )

        if edge.relation == EdgeType.SUPPORTS:
            return self._literature_hypothesis(
                edge
            )

        return self._association_hypothesis(
            edge
        )

    @staticmethod
    def _edge_confidence(
        edge,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                float(edge.confidence),
            ),
        )

    @staticmethod
    def _deduplicate(
        hypotheses: list[ResearchHypothesis],
    ) -> list[ResearchHypothesis]:
        seen: set[str] = set()
        result: list[ResearchHypothesis] = []

        for hypothesis in hypotheses:
            key = (
                hypothesis.title.casefold()
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(
                hypothesis
            )

        result.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        return result


__all__ = [
    "HypothesisGenerator",
    "ResearchHypothesis",
]
