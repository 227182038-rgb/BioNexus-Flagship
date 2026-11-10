"""Unit tests for NormalizationResolver in nexus.normalization.resolver."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.normalization.resolver import NormalizationResolver


def test_normalization_resolver_resolve() -> None:
    resolver = NormalizationResolver()
    e1 = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
        aliases=["P53"],
        cross_references={"ensembl": "ENSG00000141510"},
    )
    e2 = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        aliases=["LFS1"],
    )

    results = resolver.resolve([e1, e2])
    assert len(results) == 1
    entity = results[0]
    assert entity.canonical_id == "UNIPROT:P04637"
    assert "P53" in entity.aliases
    assert "LFS1" in entity.aliases
    assert entity.confidence > 0.0


def test_normalization_resolver_normalize_single() -> None:
    resolver = NormalizationResolver()
    entity = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
    )
    result = resolver.normalize(entity)
    assert result.entity.canonical_id == "UNIPROT:P04637"
    assert result.confidence.overall > 0.0
