"""Unit tests for IdentifierNormalizer in nexus.normalization.identifier."""

from __future__ import annotations

import pytest

from nexus.normalization.identifier import IdentifierNamespace, IdentifierNormalizer


def test_identifier_normalizer_uniprot() -> None:
    norm = IdentifierNormalizer.normalize("uniprot:P04637")
    assert norm.namespace == IdentifierNamespace.UNIPROT
    assert norm.value == "P04637"
    assert norm.canonical == "uniprot:P04637"


def test_identifier_normalizer_ensembl() -> None:
    norm = IdentifierNormalizer.normalize("ENSG00000141510")
    assert norm.namespace == IdentifierNamespace.ENSEMBL
    assert norm.value == "ENSG00000141510"


def test_identifier_normalizer_pdb() -> None:
    norm = IdentifierNormalizer.normalize("pdb:1a1u")
    assert norm.namespace == IdentifierNamespace.PDB
    assert norm.value == "1A1U"


def test_identifier_normalizer_pubmed() -> None:
    norm = IdentifierNormalizer.normalize("pmid:123456")
    assert norm.namespace == IdentifierNamespace.PUBMED
    assert norm.value == "123456"


def test_identifier_normalizer_invalid() -> None:
    with pytest.raises(ValueError):
        IdentifierNormalizer.normalize("")
