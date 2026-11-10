"""Unit tests for EntityCanonicalizer in nexus.normalization.canonicalizer."""

from __future__ import annotations

import pytest

from nexus.fusion.models import BiomedicalEntity
from nexus.normalization.canonicalizer import EntityCanonicalizer


def test_canonicalize_single_entity() -> None:
    entity = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
        aliases=["P53"],
    )
    canonicalizer = EntityCanonicalizer()
    result = canonicalizer.canonicalize(entity)

    assert result.canonical_id == "UNIPROT:P04637"
    assert result.name == "TP53"
    assert "P53" in result.aliases


def test_canonicalize_type_error() -> None:
    canonicalizer = EntityCanonicalizer()
    with pytest.raises(TypeError, match="entity must be a BiomedicalEntity"):
        canonicalizer.canonicalize("not_an_entity")  # type: ignore[arg-type]


def test_canonicalize_many() -> None:
    e1 = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
    )
    e2 = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        aliases=["P53"],
    )
    canonicalizer = EntityCanonicalizer()
    merged = canonicalizer.canonicalize_many([e1, e2])

    assert len(merged) == 1
    assert merged[0].canonical_id == "UNIPROT:P04637"
    assert "P53" in merged[0].aliases
