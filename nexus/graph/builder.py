"""Knowledge graph builder."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.normalization.identifier import (
    IdentifierNamespace,
    IdentifierNormalizer,
)

from .edge import EdgeType, GraphEdge
from .graph import KnowledgeGraph
from .literature import LiteratureLinker
from .node import GraphNode
from .relationships import RelationshipResolver
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id


class GraphBuilder:
    """Build a KnowledgeGraph from canonical BiomedicalEntity objects."""

    def __init__(
        self,
        relationship_resolver: RelationshipResolver | None = None,
        literature_linker: LiteratureLinker | None = None,
    ) -> None:
        self.logger = get_logger("graph")
        self.graph = KnowledgeGraph()

        self.relationships = (
            relationship_resolver
            or RelationshipResolver()
        )

        self.literature = (
            literature_linker
            or LiteratureLinker()
        )

    def add_entity(
        self,
        entity: BiomedicalEntity,
    ) -> None:
        """Convert one BiomedicalEntity into graph nodes and edges."""

        node_id = self._entity_id(entity)

        properties = dict(entity.metadata)

        properties["original_id"] = entity.id

        if entity.canonical_id:
            properties["canonical_id"] = entity.canonical_id

        if entity.canonical_source:
            properties["canonical_source"] = entity.canonical_source

        node = GraphNode(
            id=node_id,
            label=entity.name or entity.id,
            type=str(entity.entity_type),
            aliases=list(entity.aliases),
            properties=properties,
            confidence=entity.confidence,
        )

        if not self.graph.has_node(node_id):
            self.graph.add_node(node)
        else:
            existing = self.graph.get_node(node_id)

            if existing is not None:
                if entity.name:
                    existing.add_alias(entity.name)

                if entity.id:
                    existing.add_alias(entity.id)

                for alias in entity.aliases:
                    existing.add_alias(alias)

                existing.update_properties(properties)

                existing.confidence = max(
                    existing.confidence,
                    entity.confidence,
                )

        self._add_collection(
            node_id,
            entity.genes,
            "Gene",
            EdgeType.ASSOCIATED_WITH,
        )

        self._add_collection(
            node_id,
            entity.proteins,
            "Protein",
            EdgeType.ASSOCIATED_WITH,
        )

        self._add_collection(
            node_id,
            entity.diseases,
            "Disease",
            EdgeType.ASSOCIATED_WITH,
        )

        self._add_collection(
            node_id,
            entity.pathways,
            "Pathway",
            EdgeType.PARTICIPATES_IN,
        )

        self._add_collection(
            node_id,
            entity.structures,
            "Structure",
            EdgeType.HAS_STRUCTURE,
        )

        self._add_collection(
            node_id,
            entity.publications,
            "Publication",
            EdgeType.SUPPORTS,
        )

    def _add_collection(
        self,
        source: str,
        values: list[str],
        node_type: str,
        relation: EdgeType,
    ) -> None:
        """Add referenced entities using canonical graph identifiers."""

        for value in values:
            canonical_id = self._reference_id(
                value,
                node_type,
            )

            if canonical_id == source:
                continue

            if not self.graph.has_node(canonical_id):
                self.graph.add_node(
                    GraphNode(
                        id=canonical_id,
                        label=value,
                        type=node_type,
                        properties={
                            "original_id": value,
                            "canonical_id": canonical_id,
                        },
                    )
                )
            else:
                existing = self.graph.get_node(
                    canonical_id
                )

                if existing is not None:
                    existing.add_alias(value)

                    existing.update_properties(
                        {
                            "original_id": value,
                            "canonical_id": canonical_id,
                        }
                    )

            self.graph.add_edge(
                GraphEdge(
                    source=source,
                    target=canonical_id,
                    relation=relation,
                )
            )

    def _add_relationships(
        self,
        entities: list[BiomedicalEntity],
    ) -> None:
        """Add authoritative cross-reference relationships."""

        candidates = self.relationships.resolve(
            entities
        )

        for candidate in candidates:
            if candidate.source == candidate.target:
                continue

            if (
                not self.graph.has_node(candidate.source)
                or not self.graph.has_node(candidate.target)
            ):
                continue

            self.graph.add_edge(
                GraphEdge(
                    source=candidate.source,
                    target=candidate.target,
                    relation=candidate.relation,
                    confidence=candidate.confidence,
                    evidence=list(candidate.evidence),
                    properties={
                        "relationship_source": (
                            "authoritative_cross_reference"
                        ),
                    },
                    directed=True,
                )
            )

    def _propagate_authoritative_aliases(
        self,
        entities: list[BiomedicalEntity],
    ) -> None:
        """Propagate aliases across authoritative gene-protein mappings."""

        entity_map = {
            (
                entity.canonical_id
                or entity.id
            ): entity
            for entity in entities
            if (
                entity.canonical_id
                or entity.id
            )
        }

        candidates = self.relationships.resolve(
            entities
        )

        for candidate in candidates:
            if candidate.relation != EdgeType.ENCODES:
                continue

            source = entity_map.get(
                candidate.source
            )
            target = entity_map.get(
                candidate.target
            )

            if source is None or target is None:
                continue

            for alias in target.aliases:
                if alias and alias not in source.aliases:
                    source.aliases.append(alias)

    def _add_authoritative_structure_links(
        self,
        entities: list[BiomedicalEntity],
    ) -> None:
        """Link genes to PDB structures through authoritative UniProt mappings."""

        entity_map = {
            (
                entity.canonical_id
                or entity.id
            ): entity
            for entity in entities
            if (
                entity.canonical_id
                or entity.id
            )
        }

        candidates = self.relationships.resolve(
            entities
        )

        for candidate in candidates:
            if candidate.relation != EdgeType.ENCODES:
                continue

            gene = entity_map.get(
                candidate.source
            )

            protein = entity_map.get(
                candidate.target
            )

            if gene is None or protein is None:
                continue

            pdb_reference = protein.cross_references.get(
                "pdb"
            )

            if not pdb_reference:
                continue

            pdb_id = str(
                pdb_reference
            ).strip()

            if pdb_id.upper().startswith("PDB:"):
                pdb_id = pdb_id.split(
                    ":",
                    1,
                )[1].strip()

            pdb_id = pdb_id.upper()

            structure_id = f"pdb:{pdb_id}"

            if structure_id not in entity_map:
                continue

            if not self.graph.has_node(
                gene.canonical_id or gene.id
            ):
                continue

            if not self.graph.has_node(
                structure_id
            ):
                continue

            self.graph.add_edge(
                GraphEdge(
                    source=(
                        gene.canonical_id
                        or gene.id
                    ),
                    target=structure_id,
                    relation=EdgeType.HAS_STRUCTURE,
                    confidence=min(
                        candidate.confidence,
                        0.98,
                    ),
                    evidence=[
                        *candidate.evidence,
                        f"authoritative_xref:pdb:{pdb_id}",
                    ],
                    properties={
                        "relationship_source": (
                            "authoritative_cross_reference"
                        ),
                        "via_protein": (
                            protein.canonical_id
                            or protein.id
                        ),
                    },
                    directed=True,
                )
            )
    def _add_literature_links(
        self,
        subject: BiomedicalEntity,
        entities: list[BiomedicalEntity],
        correlation_id: str | None = None,
    ) -> None:
        """Add relevance-scored literature relationships."""

        publications = [
            entity
            for entity in entities
            if str(entity.entity_type).casefold()
            == "publication"
        ]

        links = self.literature.link(
            subject,
            publications,
            correlation_id=correlation_id,
        )

        for link in links:
            if (
                not self.graph.has_node(link.source)
                or not self.graph.has_node(link.target)
            ):
                continue

            self.graph.add_edge(
                GraphEdge(
                    source=link.source,
                    target=link.target,
                    relation=link.relation,
                    weight=link.relevance,
                    confidence=link.relevance,
                    evidence=list(link.evidence),
                    properties={
                        "relationship_source": (
                            "literature_relevance"
                        ),
                        "tier": link.tier,
                    },
                    directed=True,
                )
            )

    @staticmethod
    def _entity_id(
        entity: BiomedicalEntity,
    ) -> str:
        """Return the strongest available canonical entity identifier."""

        if entity.canonical_id:
            return entity.canonical_id

        try:
            normalized = IdentifierNormalizer.normalize(
                entity.id,
            )

            if normalized.namespace != IdentifierNamespace.UNKNOWN:
                return normalized.canonical

        except (TypeError, ValueError):
            pass

        return entity.id

    @staticmethod
    def _reference_id(
        value: str,
        node_type: str,
    ) -> str:
        """Canonicalize a graph reference using biological context."""

        try:
            normalized = IdentifierNormalizer.normalize(
                value,
            )

            if normalized.namespace != IdentifierNamespace.UNKNOWN:
                return normalized.canonical

        except (TypeError, ValueError):
            pass

        if node_type == "Publication":
            try:
                normalized = IdentifierNormalizer.normalize(
                    value,
                    IdentifierNamespace.PUBMED,
                )

                return normalized.canonical

            except (TypeError, ValueError):
                pass

        return value

    def build(
        self,
        entities: list[BiomedicalEntity],
        *,
        subject: BiomedicalEntity | None = None,
        correlation_id: str | None = None,
    ) -> KnowledgeGraph:
        """Build a fresh graph from supplied entities."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.info("[%s] Building knowledge graph from %d entities", cid, len(entities))

        try:
            self.graph.clear()

            for entity in entities:
                self.add_entity(entity)

            self._add_relationships(
                entities
            )

            self._propagate_authoritative_aliases(
                entities
            )

            self._add_authoritative_structure_links(
                entities
            )

            if subject is not None:
                self._add_literature_links(
                    subject,
                    entities,
                    correlation_id=cid,
                )

            return self.graph
        except Exception as exc:
            self.logger.error("Graph construction error [%s]: %s", cid, exc)
            raise


__all__ = [
    "GraphBuilder",
]




