"""Unit tests for AliasResolver in nexus.normalization.aliases."""

from __future__ import annotations

import pytest

from nexus.normalization.aliases import AliasResolver


def test_alias_resolver_register_and_resolve() -> None:
    resolver = AliasResolver()
    resolver.register("UNIPROT:P04637", ["P53", "TP53", "LFS1"])

    match = resolver.resolve("p53")
    assert match is not None
    assert match.canonical_id == "UNIPROT:P04637"

    match_tp53 = resolver.resolve("TP53")
    assert match_tp53 is not None
    assert match_tp53.canonical_id == "UNIPROT:P04637"

    aliases = resolver.get_aliases("UNIPROT:P04637")
    assert "P53" in aliases
    assert "TP53" in aliases


def test_alias_resolver_empty_canonical_id() -> None:
    resolver = AliasResolver()
    with pytest.raises(ValueError, match="canonical_id cannot be empty"):
        resolver.register("", ["alias"])


def test_alias_resolver_unregistered() -> None:
    resolver = AliasResolver()
    assert resolver.resolve("UNKNOWN_GENE") is None
