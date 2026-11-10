"""Unit tests for EntityResolver."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.fusion.resolver import EntityResolver
from nexus.knowledge.base import KnowledgeRecord, KnowledgeSource


def test_entity_resolver_resolve_uniprot() -> None:
    rec1 = KnowledgeRecord(
        source=KnowledgeSource.UNIPROT,
        source_id="P04637",
        title="P04637",
        content="Cellular tumor antigen p53",
        uri="https://uniprot.org/uniprot/P04637",
        metadata={
            "accession": "P04637",
            "gene_symbol": "TP53",
            "gene_synonyms": ["P53", "LFS1"],
            "cross_references": {"ensembl": "ENSG00000141510", "pdb": "1A1U"},
        },
    )
    resolver = EntityResolver()
    entities = resolver.resolve([rec1])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_id == "UNIPROT:P04637"
    assert "P04637" in entity.proteins
    assert "TP53" in entity.aliases
    assert "P53" in entity.aliases
    assert entity.cross_references.get("ensembl") == "ENSG00000141510"


def test_entity_resolver_resolve_ensembl() -> None:
    rec = KnowledgeRecord(
        source=KnowledgeSource.ENSEMBL,
        source_id="ENSG00000141510",
        title="ENSG00000141510",
        content="Tumor protein p53 gene",
        uri="https://ensembl.org/id/ENSG00000141510",
        metadata={"ensembl_id": "ENSG00000141510", "symbol": "TP53"},
    )
    resolver = EntityResolver()
    entities = resolver.resolve([rec])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_id == "ENSEMBL:ENSG00000141510"
    assert "ENSG00000141510" in entity.genes
    assert entity.name == "TP53"


def test_entity_resolver_resolve_pdb() -> None:
    rec = KnowledgeRecord(
        source=KnowledgeSource.PDB,
        source_id="1A1U",
        title="Structure 1A1U",
        content="PDB structure",
        uri="https://rcsb.org/structure/1A1U",
        metadata={"pdb_id": "1A1U", "method": "X-RAY"},
    )
    resolver = EntityResolver()
    entities = resolver.resolve([rec])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_id == "PDB:1A1U"
    assert "1A1U" in entity.structures


def test_entity_resolver_resolve_go() -> None:
    rec = KnowledgeRecord(
        source=KnowledgeSource.GO,
        source_id="GO:0006915",
        title="apoptotic process",
        content="Gene ontology term",
        uri="https://quickgo.org/term/GO:0006915",
        metadata={"go_id": "GO:0006915", "name": "apoptotic process"},
    )
    resolver = EntityResolver()
    entities = resolver.resolve([rec])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_id == "go:GO:0006915"
    assert "GO:0006915" in entity.ontology_terms


def test_entity_resolver_resolve_pubmed() -> None:
    rec = KnowledgeRecord(
        source=KnowledgeSource.PUBMED,
        source_id="123456",
        title="BRCA1 and p53 research",
        content="Abstract content",
        uri="https://pubmed.ncbi.nlm.nih.gov/123456/",
        metadata={"pmid": "123456", "mesh_terms": ["Tumor Suppressor Protein p53"]},
    )
    resolver = EntityResolver()
    entities = resolver.resolve([rec])

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_id == "pubmed:123456"
    assert "https://pubmed.ncbi.nlm.nih.gov/123456/" in entity.publications
