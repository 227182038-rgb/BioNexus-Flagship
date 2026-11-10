"""Biomedical ontology normalization.

Provides deterministic ontology-term normalization and local mapping support.
External ontology services should be integrated through adapters rather than
directly from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class OntologyNamespace(str, Enum):
    """Supported ontology namespaces."""

    GO = "GO"
    MONDO = "MONDO"
    DOID = "DOID"
    SO = "SO"
    CHEBI = "CHEBI"
    REACTOME = "REACTOME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class OntologyTerm:
    """Canonical representation of an ontology term."""

    namespace: OntologyNamespace
    identifier: str
    original: str
    label: str | None = None

    @property
    def canonical_id(self) -> str:
        """Return the namespace-qualified ontology identifier."""
        return f"{self.namespace.value}:{self.identifier}"


@dataclass(slots=True)
class OntologyNormalizer:
    """Normalize ontology identifiers and maintain local mappings."""

    _mappings: dict[str, OntologyTerm] = field(default_factory=dict)

    _PREFIXES = {
        "go": OntologyNamespace.GO,
        "mondo": OntologyNamespace.MONDO,
        "doid": OntologyNamespace.DOID,
        "so": OntologyNamespace.SO,
        "chebi": OntologyNamespace.CHEBI,
        "reactome": OntologyNamespace.REACTOME,
    }

    def normalize(self, term: str) -> OntologyTerm:
        """Normalize an ontology identifier."""
        if not isinstance(term, str):
            raise TypeError("term must be a string")

        original = term
        value = term.strip()

        if not value:
            raise ValueError("ontology term cannot be empty")

        namespace, identifier = self._parse(value)

        canonical_key = f"{namespace.value}:{identifier}"

        mapped = self._mappings.get(canonical_key)

        if mapped is not None:
            return OntologyTerm(
                namespace=mapped.namespace,
                identifier=mapped.identifier,
                original=original,
                label=mapped.label,
            )

        return OntologyTerm(
            namespace=namespace,
            identifier=identifier,
            original=original,
        )

    def canonicalize(self, term: str) -> str:
        """Return the canonical ontology identifier."""
        return self.normalize(term).canonical_id

    def detect_namespace(self, term: str) -> OntologyNamespace:
        """Detect the ontology namespace."""
        return self.normalize(term).namespace

    def register_mapping(
        self,
        term: str,
        *,
        label: str | None = None,
    ) -> OntologyTerm:
        """Register a locally known ontology term."""
        normalized = self.normalize(term)

        canonical = OntologyTerm(
            namespace=normalized.namespace,
            identifier=normalized.identifier,
            original=normalized.original,
            label=label or normalized.label,
        )

        self._mappings[canonical.canonical_id] = canonical

        return canonical

    def resolve(self, term: str) -> OntologyTerm | None:
        """Resolve a locally registered ontology term."""
        normalized = self.normalize(term)

        return self._mappings.get(normalized.canonical_id)

    def _parse(
        self,
        value: str,
    ) -> tuple[OntologyNamespace, str]:
        if ":" not in value:
            raise ValueError(
                f"Ontology term must contain a namespace: {value}"
            )

        prefix, identifier = value.split(":", 1)

        namespace = self._PREFIXES.get(
            prefix.strip().lower()
        )

        if namespace is None:
            raise ValueError(
                f"Unsupported ontology namespace: {prefix}"
            )

        identifier = identifier.strip()

        if not identifier:
            raise ValueError(
                "Ontology identifier cannot be empty"
            )

        identifier = identifier.upper()

        self._validate(namespace, identifier)

        return namespace, identifier

    @staticmethod
    def _validate(
        namespace: OntologyNamespace,
        identifier: str,
    ) -> None:
        if namespace == OntologyNamespace.GO:
            if not re.fullmatch(r"\d+", identifier):
                raise ValueError(
                    f"Invalid GO identifier: {identifier}"
                )

        elif namespace == OntologyNamespace.CHEBI:
            if not re.fullmatch(r"\d+", identifier):
                raise ValueError(
                    f"Invalid ChEBI identifier: {identifier}"
                )

        elif namespace in {
            OntologyNamespace.MONDO,
            OntologyNamespace.DOID,
            OntologyNamespace.SO,
        }:
            if not re.fullmatch(
                r"[A-Z0-9._-]+",
                identifier,
            ):
                raise ValueError(
                    f"Invalid {namespace.value} identifier: "
                    f"{identifier}"
                )

        elif namespace == OntologyNamespace.REACTOME:
            if not re.fullmatch(
                r"[A-Z0-9._-]+",
                identifier,
            ):
                raise ValueError(
                    f"Invalid Reactome identifier: {identifier}"
                )


__all__ = [
    "OntologyNamespace",
    "OntologyNormalizer",
    "OntologyTerm",
]
