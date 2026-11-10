"""Biomedical analysis workflow."""

from __future__ import annotations

from nexus.fusion.engine import FusionEngine
from nexus.fusion.models import BiomedicalEntity
from nexus.fusion.subject import SubjectResolver
from nexus.graph.builder import GraphBuilder
from nexus.graph.edge import EdgeType
from nexus.knowledge.ncbi import NCBIClient
from nexus.knowledge.pubmed import PubMedClient
from nexus.knowledge.uniprot import UniProtClient
from nexus.knowledge.pdb import PDBClient
from nexus.knowledge.go import GOClient
from nexus.knowledge.ensembl import EnsemblClient
from nexus.reasoning.engine import ReasoningEngine
from nexus.reasoning.evidence import EvidenceManager
from nexus.reasoning.evidence_bridge import (
    populate_subject_evidence,
)
from nexus.pipeline.context import ScientificAnalysisContext
from nexus.pipeline.report import ScientificReport
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id

from .workflow import Workflow


class AnalyzeBiomedicalEntityWorkflow(
    Workflow[str, ScientificReport]
):
    """End-to-end biomedical analysis workflow."""

    def __init__(self) -> None:
        self.logger = get_logger("workflow")
        self.clients = [
            PubMedClient(),
            NCBIClient(),
            UniProtClient(),
            PDBClient(),
            GOClient(),
            EnsemblClient(),
        ]

        self.fusion = FusionEngine(
            self.clients
        )

        self.builder = GraphBuilder()

        self.subject_resolver = SubjectResolver(
            self.fusion._router
        )

    async def run_context(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> ScientificAnalysisContext:
        """Execute analysis and retain the complete scientific context."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.info("[%s] Starting biomedical entity workflow for query: %s", cid, query)

        entities = await self.fusion.search(
            query,
            correlation_id=cid,
        )

        if not entities:
            self.logger.warning("[%s] No entities retrieved for query: %s", cid, query)
            report = ScientificReport(
                query=query,
                ranked_entities=[],
                explanation=None,
                contradictions=[],
                hypotheses=[],
                metadata={"correlation_id": cid},
            )

            return ScientificAnalysisContext(
                query=query,
                subject=None,
                entities=[],
                graph=None,
                evidence=[],
                report=report,
                correlation_id=cid,
            )

        selection = self.subject_resolver.select(
            query,
            entities,
        )

        if selection is None:
            self.logger.warning("[%s] No primary subject selected for query: %s", cid, query)
            report = ScientificReport(
                query=query,
                ranked_entities=[],
                explanation=None,
                contradictions=[],
                hypotheses=[],
                metadata={"correlation_id": cid},
            )

            return ScientificAnalysisContext(
                query=query,
                subject=None,
                entities=entities,
                graph=None,
                evidence=[],
                report=report,
                correlation_id=cid,
            )

        self._enrich_subject_aliases(
            selection.entity,
            entities,
        )

        graph = self.builder.build(
            entities,
            subject=selection.entity,
            correlation_id=cid,
        )

        evidence_manager = EvidenceManager()

        populate_subject_evidence(
            selection.entity,
            entities,
            graph,
            evidence_manager,
            correlation_id=cid,
        )

        reasoning = ReasoningEngine(
            graph
        )

        reasoning_report = reasoning.analyze(
            selection.entity,
            evidence_manager,
            query=query,
            correlation_id=cid,
        )

        report = ScientificReport(
            query=query,
            ranked_entities=reasoning_report.ranked_entities,
            explanation=reasoning_report.explanation,
            contradictions=reasoning_report.contradictions,
            hypotheses=reasoning_report.hypotheses,
            scientific_confidence=(
                reasoning_report.scientific_confidence
            ),
            metadata={
                "correlation_id": cid,
                "subject_id": (
                    selection.entity.canonical_id
                    or selection.entity.id
                ),
                "subject_name": selection.entity.name,
                "subject_type": str(
                    selection.entity.entity_type
                ),
                "evidence_count": str(
                    len(evidence_manager.all())
                ),
            },
        )

        return ScientificAnalysisContext(
            query=query,
            subject=selection.entity,
            entities=entities,
            graph=graph,
            evidence=evidence_manager.all(),
            report=report,
            correlation_id=cid,
        )

    def _enrich_subject_aliases(
        self,
        subject: BiomedicalEntity,
        entities: list[BiomedicalEntity],
    ) -> None:
        """Propagate biological synonyms from authoritative ENCODES partners."""

        subject_id = (
            subject.canonical_id
            or subject.id
        )

        candidates = self.builder.relationships.resolve(
            entities
        )

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

        for candidate in candidates:
            if candidate.relation != EdgeType.ENCODES:
                continue

            partner_id = None
            if candidate.source == subject_id:
                partner_id = candidate.target
            elif candidate.target == subject_id:
                partner_id = candidate.source

            if partner_id is None:
                continue

            partner = entity_map.get(partner_id)

            if partner is None:
                continue

            for alias in partner.aliases:
                if not alias:
                    continue

                # Keep biological synonyms; don't propagate database accessions.
                if alias.casefold() == partner.id.casefold():
                    continue

                if alias not in subject.aliases:
                    subject.aliases.append(alias)
    async def run(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> ScientificReport:
        """Execute the complete biomedical analysis workflow."""

        context = await self.run_context(
            query,
            correlation_id=correlation_id,
        )

        assert context.report is not None

        return context.report


__all__ = [
    "AnalyzeBiomedicalEntityWorkflow",
]

