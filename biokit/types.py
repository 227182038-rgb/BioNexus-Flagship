"""Shared type aliases and protocols for BioKit 2.0.

This module centralises type aliases used across BioKit so that public
signatures stay consistent and tooling (mypy, IDEs) has one canonical
reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

# Re-exported for convenience.
PathLike = str | Path  # repo convention, see biokit/utils/io_utils.py

# A nucleotide or protein sequence. We keep this as ``str`` rather than a
# custom class because Biopython and the wider ecosystem use ``str``.
Sequence = str

# Genotype alias: a 2-tuple of allele labels.
Genotype = tuple[str, str]

# Generic type variable used by many modules.
T = TypeVar("T")


@runtime_checkable
class Parseable(Protocol):
    """Protocol for objects that can be parsed from a string."""

    @classmethod
    def parse(cls, text: str) -> Parseable:
        """Parse from a string."""
        ...


@runtime_checkable
class Renderable(Protocol):
    """Protocol for objects that can be rendered to a string."""

    def render(self) -> str:
        """Render to a string."""
        ...


__all__ = [
    "Genotype",
    "Parseable",
    "PathLike",
    "Renderable",
    "Sequence",
    "T",
]
