"""Canonicalization of biomedical entities.

Transforms resolved BiomedicalEntity objects into deterministic canonical
representations while preserving source provenance and biomedical metadata.

This module performs no network access and does not attempt to invent
authoritative identifiers that are not present in the supplied entities.
"""

from __future__ import annotations

from dataclasses import replace

from nexus.fusion.models import BiomedicalEntity, Evidence

from .aliases import AliasResolver
from .identifier import IdentifierNamespace, IdentifierNormalizer


class CanonicalizationError(ValueError):
    """Raised when an entity cannot be safely canonicalized."""


class EntityCanonicalizer:
    """Canonicalize and merge biomedical entities.

    The canonicalizer operates on entities already produced by the fusion
    layer. It deliberately does not perform external lookups.
    """

    def __init__(
        self,
        alias_resolver: AliasResolver | None = None,
    ) -> None:
        self._aliases = alias_resolver or AliasResolver()

    def canonicalize(
        self,
        entity: BiomedicalEntity,
    ) -> BiomedicalEntity:
        """Return a canonical copy of one biomedical entity."""
        if not isinstance(entity, BiomedicalEntity):
            raise TypeError("entity must be a BiomedicalEntity")

        result = replace(entity)

        result.aliases = list(entity.aliases)
        result.genes = list(entity.genes)
        result.proteins = list(entity.proteins)
        result.structures = list(entity.structures)
        result.ontology_terms = list(entity.ontology_terms)
        result.diseases = list(entity.diseases)
        result.pathways = list(entity.pathways)
        result.publications = list(entity.publications)
        result.sequences = dict(entity.sequences)
        result.cross_references = dict(entity.cross_references)
        result.metadata = dict(entity.metadata)
        result.evidence = list(entity.evidence)

        normalized = self._normalize_identity(entity)

        if normalized is not None:
            result.canonical_id = normalized.canonical
            result.canonical_source = normalized.namespace.value

        result.aliases = self._unique(
            result.aliases
            + ([entity.name] if entity.name else [])
            + ([entity.id] if entity.id else [])
        )

        return result

    def canonicalize_many(
        self,
        entities: list[BiomedicalEntity],
    ) -> list[BiomedicalEntity]:
        """Canonicalize entities and merge entities sharing an identity."""
        canonical_entities: dict[str, BiomedicalEntity] = {}

        for entity in entities:
            canonical = self.canonicalize(entity)

            key = self._identity_key(canonical)

            existing = canonical_entities.get(key)

            if existing is None:
                canonical_entities[key] = canonical
            else:
                self._merge(existing, canonical)

        return list(canonical_entities.values())

    def merge(
        self,
        target: BiomedicalEntity,
        incoming: BiomedicalEntity,
    ) -> BiomedicalEntity:
        """Merge an incoming entity into a canonical target.

        The target is mutated using the existing BiomedicalEntity.merge()
        contract and then re-canonicalized.
        """
        if not isinstance(target, BiomedicalEntity):
            raise TypeError("target must be a BiomedicalEntity")

        if not isinstance(incoming, BiomedicalEntity):
            raise TypeError("incoming must be a BiomedicalEntity")

        target.merge(incoming)

        normalized = self._normalize_identity(target)

        if normalized is not None:
            target.canonical_id = normalized.canonical
            target.canonical_source = normalized.namespace.value

        target.aliases = self._unique(target.aliases)

        return target

    def _normalize_identity(self, entity: BiomedicalEntity):
        """Normalize the strongest available entity identifier."""
        candidates: list[str] = []

        if entity.canonical_id:
            candidates.append(entity.canonical_id)

        if entity.id:
            candidates.append(entity.id)

        candidates.extend(
            value
            for value in entity.cross_references.values()
            if value
        )

        for candidate in candidates:
            try:
                normalized = IdentifierNormalizer.normalize(candidate)

            except (TypeError, ValueError):
                continue

            if normalized.namespace != IdentifierNamespace.UNKNOWN:
                return normalized

        return None

    @staticmethod
    def _identity_key(entity: BiomedicalEntity) -> str:
        """Return a deterministic grouping key."""
        if entity.canonical_id:
            return entity.canonical_id.casefold()

        if entity.id:
            return entity.id.strip().casefold()

        if entity.name:
            return entity.name.strip().casefold()

        raise CanonicalizationError(
            "Entity has no usable identity"
        )

    @staticmethod
    def _merge(
        target: BiomedicalEntity,
        incoming: BiomedicalEntity,
    ) -> None:
        """Merge entities using the domain model's merge semantics."""
        target.merge(incoming)

        if (
            target.canonical_id is None
            and incoming.canonical_id is not None
        ):
            target.canonical_id = incoming.canonical_id

        if (
            target.canonical_source is None
            and incoming.canonical_source is not None
        ):
            target.canonical_source = incoming.canonical_source

        target.aliases = EntityCanonicalizer._unique(
            target.aliases
        )

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """Deduplicate strings while preserving their original order."""
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            if not value:
                continue

            key = value.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result


__all__ = [
    "CanonicalizationError",
    "EntityCanonicalizer",
]
