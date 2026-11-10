"""Unit tests for NormalizationResolver service wrappers."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.normalization.resolver import NormalizationResolver


def test_normalization_service_alias_registration() -> None:
    resolver = NormalizationResolver()
    resolver.register_aliases("UNIPROT:P04637", ["P53", "TP53"])
    resolver.register_ontology_term("GO:0006915", label="apoptotic process")

    e = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        aliases=["P53"],
        ontology_terms=["GO:0006915"],
    )
    res = resolver.normalize(e)
    assert res.entity.canonical_id == "UNIPROT:P04637"
    assert "GO:0006915" in res.entity.ontology_terms
