"""Canonical biomedical entity models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entity_type import EntityType


@dataclass(slots=True)
class Evidence:
    """Represents supporting evidence for a biomedical entity."""

    source: str
    source_id: str
    confidence: float = 1.0

    url: str | None = None
    version: str | None = None
    retrieved_at: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BiomedicalEntity:
    """Canonical biomedical entity."""

    # Original identifier from retrieval source
    id: str

    # Canonical identifier after normalization
    canonical_id: str | None = None
    canonical_source: str | None = None

    name: str = ""

    entity_type: EntityType | str = EntityType.UNKNOWN

    description: str = ""

    aliases: list[str] = field(default_factory=list)

    genes: list[str] = field(default_factory=list)
    proteins: list[str] = field(default_factory=list)
    structures: list[str] = field(default_factory=list)

    ontology_terms: list[str] = field(default_factory=list)

    diseases: list[str] = field(default_factory=list)
    pathways: list[str] = field(default_factory=list)
    publications: list[str] = field(default_factory=list)

    sequences: dict[str, str] = field(default_factory=dict)

    cross_references: dict[str, str] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    evidence: list[Evidence] = field(default_factory=list)

    confidence: float = 0.0

    @property
    def is_canonical(self) -> bool:
        return self.canonical_id is not None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def has_cross_references(self) -> bool:
        return bool(self.cross_references)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def merge_aliases(self, aliases: list[str]) -> None:
        for alias in aliases:
            if alias and alias not in self.aliases:
                self.aliases.append(alias)

    def merge_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def merge(self, other: "BiomedicalEntity") -> None:
        """
        Merge another BiomedicalEntity into this one.
        """

        self.merge_aliases(other.aliases)

        self.merge_metadata(other.metadata)

        self.confidence = max(self.confidence, other.confidence)

        self.cross_references.update(other.cross_references)

        self.sequences.update(other.sequences)

        for attr in (
            "genes",
            "proteins",
            "structures",
            "ontology_terms",
            "diseases",
            "pathways",
            "publications",
        ):
            current = getattr(self, attr)
            incoming = getattr(other, attr)

            for item in incoming:
                if item not in current:
                    current.append(item)

        for ev in other.evidence:
            if ev not in self.evidence:
                self.evidence.append(ev)