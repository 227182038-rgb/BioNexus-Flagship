"""Bridge graph/fusion evidence into reasoning evidence."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.graph.graph import KnowledgeGraph
from nexus.reasoning.evidence import Evidence, EvidenceManager
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id

_logger = get_logger("evidence_bridge")


def populate_evidence_manager(
    entity: BiomedicalEntity,
    manager: EvidenceManager,
    correlation_id: str | None = None,
) -> EvidenceManager:
    """Add direct fusion evidence from one biomedical entity."""

    for item in entity.evidence:
        _add_fusion_evidence(
            item,
            entity,
            manager,
        )

    return manager


def populate_subject_evidence(
    subject: BiomedicalEntity,
    entities: list[BiomedicalEntity],
    graph: KnowledgeGraph,
    manager: EvidenceManager,
    correlation_id: str | None = None,
) -> EvidenceManager:
    """Aggregate evidence surrounding a canonical subject.

    Evidence comes from:
    - the selected subject itself;
    - entities directly connected to it;
    - graph-edge provenance and confidence.

    This preserves provenance while allowing the reasoning layer to see the
    complete local scientific context.
    """

    subject_id = (
        subject.canonical_id
        or subject.id
    )

    entity_map: dict[str, BiomedicalEntity] = {}

    for entity in entities:
        entity_id = (
            entity.canonical_id
            or entity.id
        )

        if entity_id:
            entity_map[entity_id] = entity

    # 1. Direct subject evidence.
    populate_evidence_manager(
        subject,
        manager,
    )

    # 2. Evidence from directly connected entities.
    connected_ids: set[str] = set()

    for edge in graph.outgoing_edges(subject_id):
        connected_ids.add(edge.target)

    for edge in graph.incoming_edges(subject_id):
        connected_ids.add(edge.source)

    for connected_id in connected_ids:
        entity = entity_map.get(connected_id)

        if entity is None:
            continue

        for item in entity.evidence:
            _add_fusion_evidence(
                item,
                entity,
                manager,
            )

    # 3. Edge-level provenance becomes reasoning evidence too.
    for edge in graph.outgoing_edges(subject_id):
        _add_edge_evidence(
            edge,
            manager,
        )

    for edge in graph.incoming_edges(subject_id):
        _add_edge_evidence(
            edge,
            manager,
        )

    return manager


def _add_fusion_evidence(
    item,
    entity: BiomedicalEntity,
    manager: EvidenceManager,
) -> None:
    """Convert fusion-layer evidence into reasoning evidence."""

    title = str(
        item.metadata.get(
            "title",
            item.metadata.get(
                "name",
                entity.name or item.source_id,
            ),
        )
    )

    metadata = {
        str(key): str(value)
        for key, value in item.metadata.items()
    }

    if item.url:
        metadata["url"] = item.url

    if item.version:
        metadata["version"] = item.version

    if item.retrieved_at:
        metadata["retrieved_at"] = item.retrieved_at

    manager.add(
        Evidence(
            source=item.source,
            identifier=item.source_id,
            title=title,
            confidence=max(
                0.0,
                min(
                    1.0,
                    float(item.confidence),
                ),
            ),
            metadata=metadata,
        )
    )


def _add_edge_evidence(
    edge,
    manager: EvidenceManager,
) -> None:
    """Convert graph relationship provenance into reasoning evidence."""

    if not edge.evidence:
        return

    for evidence_value in edge.evidence:
        manager.add(
            Evidence(
                source="graph",
                identifier=(
                    f"{edge.source}->{edge.target}"
                ),
                title=(
                    f"{edge.relation.value}: "
                    f"{edge.source} -> {edge.target}"
                ),
                confidence=max(
                    0.0,
                    min(
                        1.0,
                        float(edge.confidence),
                    ),
                ),
                metadata={
                    "claim": edge.relation.value,
                    "evidence": evidence_value,
                    "source_node": edge.source,
                    "target_node": edge.target,
                },
            )
        )


__all__ = [
    "populate_evidence_manager",
    "populate_subject_evidence",
]
