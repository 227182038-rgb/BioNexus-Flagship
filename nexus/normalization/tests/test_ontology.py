"""Unit tests for OntologyNormalizer in nexus.normalization.ontology."""

from __future__ import annotations

import pytest

from nexus.normalization.ontology import OntologyNamespace, OntologyNormalizer


def test_ontology_normalizer_go() -> None:
    norm = OntologyNormalizer()
    term = norm.normalize("go:0006915")
    assert term.namespace == OntologyNamespace.GO
    assert term.identifier == "0006915"
    assert term.canonical_id == "GO:0006915"


def test_ontology_normalizer_type_error() -> None:
    norm = OntologyNormalizer()
    with pytest.raises(TypeError, match="term must be a string"):
        norm.normalize(123)  # type: ignore[arg-type]


def test_ontology_normalizer_register_mapping() -> None:
    norm = OntologyNormalizer()
    norm.register_mapping("GO:0006915", label="apoptotic process")
    term = norm.normalize("GO:0006915")
    assert term.label == "apoptotic process"
