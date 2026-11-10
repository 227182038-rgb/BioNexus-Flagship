"""Unit tests for BiomedicalEntity and Evidence models in nexus.fusion.models."""

from __future__ import annotations

from nexus.fusion.entity_type import EntityType
from nexus.fusion.models import BiomedicalEntity, Evidence


def test_evidence_instantiation() -> None:
    ev = Evidence(
        source="uniprot",
        source_id="P04637",
        confidence=0.98,
        url="https://uniprot.org/uniprot/P04637",
        metadata={"key": "val"},
    )
    assert ev.source == "uniprot"
    assert ev.source_id == "P04637"
    assert ev.confidence == 0.98
    assert ev.url == "https://uniprot.org/uniprot/P04637"
    assert ev.metadata == {"key": "val"}


def test_biomedical_entity_merge() -> None:
    e1 = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
        entity_type=EntityType.PROTEIN,
        aliases=["P53"],
        proteins=["P04637"],
        confidence=0.9,
    )
    e2 = BiomedicalEntity(
        id="P04637",
        aliases=["LFS1"],
        genes=["ENSG00000141510"],
        structures=["1A1U"],
        confidence=0.95,
        cross_references={"ensembl": "ENSG00000141510"},
    )

    e1.merge(e2)

    assert "P53" in e1.aliases
    assert "LFS1" in e1.aliases
    assert "ENSG00000141510" in e1.genes
    assert "1A1U" in e1.structures
    assert e1.confidence == 0.95
    assert e1.cross_references.get("ensembl") == "ENSG00000141510"


def test_biomedical_entity_properties() -> None:
    e = BiomedicalEntity(id="test_id")
    assert not e.is_canonical
    assert not e.has_evidence
    assert not e.has_cross_references

    e.canonical_id = "test:test_id"
    assert e.is_canonical

    e.add_evidence(Evidence(source="test", source_id="test_id"))
    assert e.has_evidence

    e.cross_references["xref"] = "123"
    assert e.has_cross_references
