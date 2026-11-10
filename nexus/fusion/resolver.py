"""Entity resolution for biomedical knowledge fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nexus.fusion.models import BiomedicalEntity, Evidence
from nexus.fusion.entity_type import EntityType
from nexus.knowledge.base import KnowledgeRecord, KnowledgeSource


class EntityResolver:
    """Resolve heterogeneous knowledge records into biomedical entities."""

    def resolve(
        self,
        records: list[KnowledgeRecord],
    ) -> list[BiomedicalEntity]:
        groups: dict[str, list[KnowledgeRecord]] = defaultdict(list)

        for record in records:
            key = self._entity_key(record)
            groups[key].append(record)

        return [
            self._merge(group)
            for group in groups.values()
        ]

    def _entity_key(
        self,
        record: KnowledgeRecord,
    ) -> str:
        """Return a source-aware identity key."""

        metadata = record.metadata

        if record.source == KnowledgeSource.PUBMED:
            pmid = metadata.get("pmid") or record.source_id
            return f"pubmed:{pmid}"

        if record.source == KnowledgeSource.UNIPROT:
            accession = (
                metadata.get("accession")
                or record.source_id
            )
            return f"uniprot:{accession}".upper()

        if record.source == KnowledgeSource.PDB:
            pdb_id = (
                metadata.get("pdb_id")
                or record.source_id
            )
            return f"pdb:{pdb_id}".upper()

        if record.source == KnowledgeSource.GO:
            go_id = (
                metadata.get("go_id")
                or record.source_id
            )
            go_id = str(go_id).upper()

            if not go_id.startswith("GO:"):
                go_id = f"GO:{go_id}"

            return f"go:{go_id}"

        if record.source == KnowledgeSource.ENSEMBL:
            ensembl_id = (
                metadata.get("ensembl_id")
                or record.source_id
            )
            return f"ensembl:{ensembl_id}".upper()

        if record.source == KnowledgeSource.NCBI:
            database = str(
                metadata.get("db")
                or "unknown"
            ).lower()

            return (
                f"ncbi:{database}:"
                f"{record.source_id}"
            ).upper()

        return (
            f"{record.source.value}:"
            f"{record.source_id}"
        ).lower()

    def _merge(
        self,
        records: list[KnowledgeRecord],
    ) -> BiomedicalEntity:
        """Merge records while preserving biological source semantics."""

        primary = records[0]

        entity_type = self._entity_type(primary)

        canonical_id = self._canonical_id(primary)

        entity = BiomedicalEntity(
            id=primary.source_id,
            canonical_id=canonical_id,
            canonical_source=primary.source.value,
            name=self._entity_name(primary),
            entity_type=entity_type,
            description=primary.content,
            confidence=0.0,
        )

        for record in records:
            self._merge_record(
                entity,
                record,
            )

        entity.publications = sorted(
            set(entity.publications)
        )

        return entity

    def _merge_record(
        self,
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Merge one knowledge record into an entity."""

        metadata = record.metadata

        entity.merge_metadata(metadata)

        entity.add_evidence(
            Evidence(
                source=record.source.value,
                source_id=record.source_id,
                confidence=1.0,
                url=record.uri,
                metadata=metadata,
            )
        )

        entity.cross_references[
            record.source.value
        ] = self._canonical_id(record)

        if record.source == KnowledgeSource.PUBMED:
            self._merge_publication(
                entity,
                record,
            )

        elif record.source == KnowledgeSource.NCBI:
            self._merge_ncbi(
                entity,
                record,
            )

        elif record.source == KnowledgeSource.UNIPROT:
            self._merge_uniprot(
                entity,
                record,
            )

        elif record.source == KnowledgeSource.PDB:
            self._merge_pdb(
                entity,
                record,
            )

        elif record.source == KnowledgeSource.GO:
            self._merge_go(
                entity,
                record,
            )

        elif record.source == KnowledgeSource.ENSEMBL:
            self._merge_ensembl(
                entity,
                record,
            )

    @staticmethod
    def _merge_publication(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach publication provenance without misclassifying it."""

        if record.uri not in entity.publications:
            entity.publications.append(record.uri)

    @staticmethod
    def _merge_ncbi(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach NCBI sequence provenance."""

        database = str(
            record.metadata.get(
                "db",
                "unknown",
            )
        )

        entity.sequences[
            f"ncbi:{database}:{record.source_id}"
        ] = record.content

    @staticmethod
    def _merge_uniprot(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach UniProt identity and authoritative gene mappings."""

        accession = str(
            record.metadata.get(
                "accession",
                record.source_id,
            )
        )

        if accession not in entity.proteins:
            entity.proteins.append(accession)

        gene_symbol = record.metadata.get(
            "gene_symbol"
        )

        if gene_symbol:
            gene_symbol = str(gene_symbol)

            if gene_symbol not in entity.aliases:
                entity.aliases.append(gene_symbol)

        for synonym in record.metadata.get(
            "gene_synonyms",
            [],
        ):
            synonym = str(synonym)

            if synonym not in entity.aliases:
                entity.aliases.append(synonym)

        cross_references = record.metadata.get(
            "cross_references",
            {},
        )

        if isinstance(cross_references, dict):
            for database, identifier in cross_references.items():
                if not database or not identifier:
                    continue

                entity.cross_references[
                    str(database).lower()
                ] = str(identifier)

    @staticmethod
    def _merge_pdb(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach PDB structural identity."""

        pdb_id = str(
            record.metadata.get(
                "pdb_id",
                record.source_id,
            )
        ).upper()

        if pdb_id not in entity.structures:
            entity.structures.append(pdb_id)

    @staticmethod
    def _merge_go(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach Gene Ontology annotation."""

        go_id = str(
            record.metadata.get(
                "go_id",
                record.source_id,
            )
        ).upper()

        if not go_id.startswith("GO:"):
            go_id = f"GO:{go_id}"

        if go_id not in entity.ontology_terms:
            entity.ontology_terms.append(go_id)

    @staticmethod
    def _merge_ensembl(
        entity: BiomedicalEntity,
        record: KnowledgeRecord,
    ) -> None:
        """Attach Ensembl gene identity."""

        ensembl_id = str(
            record.metadata.get(
                "ensembl_id",
                record.source_id,
            )
        ).upper()

        if ensembl_id not in entity.genes:
            entity.genes.append(ensembl_id)

        symbol = record.metadata.get("symbol")

        if symbol:
            symbol = str(symbol)

            if symbol not in entity.aliases:
                entity.aliases.append(symbol)

    @staticmethod
    def _entity_type(
        record: KnowledgeRecord,
    ) -> EntityType:
        """Map knowledge-source semantics to EntityType."""

        mapping = {
            KnowledgeSource.PUBMED: EntityType.PUBLICATION,
            KnowledgeSource.NCBI: EntityType.UNKNOWN,
            KnowledgeSource.UNIPROT: EntityType.PROTEIN,
            KnowledgeSource.PDB: EntityType.STRUCTURE,
            KnowledgeSource.GO: EntityType.ONTOLOGY,
            KnowledgeSource.ENSEMBL: EntityType.GENE,
        }

        return mapping.get(
            record.source,
            EntityType.UNKNOWN,
        )

    @staticmethod
    def _canonical_id(
        record: KnowledgeRecord,
    ) -> str:
        """Build a source-specific canonical identifier."""

        metadata: dict[str, Any] = record.metadata

        if record.source == KnowledgeSource.PUBMED:
            return f"pubmed:{metadata.get('pmid', record.source_id)}"

        if record.source == KnowledgeSource.NCBI:
            database = str(
                metadata.get(
                    "db",
                    "unknown",
                )
            ).lower()

            return (
                f"ncbi:{database}:"
                f"{record.source_id}"
            ).lower()

        if record.source == KnowledgeSource.UNIPROT:
            return (
                "uniprot:"
                f"{metadata.get('accession', record.source_id)}"
            ).upper()

        if record.source == KnowledgeSource.PDB:
            return (
                "pdb:"
                f"{metadata.get('pdb_id', record.source_id)}"
            ).upper()

        if record.source == KnowledgeSource.GO:
            go_id = str(
                metadata.get(
                    "go_id",
                    record.source_id,
                )
            ).upper()

            if not go_id.startswith("GO:"):
                go_id = f"GO:{go_id}"

            return f"go:{go_id}"

        if record.source == KnowledgeSource.ENSEMBL:
            return (
                "ensembl:"
                f"{metadata.get('ensembl_id', record.source_id)}"
            ).upper()

        return (
            f"{record.source.value}:"
            f"{record.source_id}"
        ).lower()

    @staticmethod
    def _entity_name(
        record: KnowledgeRecord,
    ) -> str:
        """Select the most useful human-readable entity name."""

        metadata = record.metadata

        if record.source == KnowledgeSource.ENSEMBL:
            symbol = metadata.get("symbol")

            if symbol:
                return str(symbol)

        if record.source == KnowledgeSource.UNIPROT:
            accession = metadata.get("accession")

            if accession:
                return str(accession)

        if record.source == KnowledgeSource.PDB:
            return str(
                metadata.get(
                    "pdb_id",
                    record.title,
                )
            )

        if record.source == KnowledgeSource.GO:
            return str(
                metadata.get(
                    "name",
                    record.title,
                )
            )

        return record.title or record.source_id


__all__ = [
    "EntityResolver",
]

