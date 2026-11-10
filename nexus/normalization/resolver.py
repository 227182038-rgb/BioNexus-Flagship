"""Biomedical entity normalization resolver.

Coordinates identifier normalization, alias resolution, canonicalization,
ontology normalization, and confidence scoring into one deterministic
normalization pipeline.

This layer performs no network access.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.fusion.models import BiomedicalEntity
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id

from .aliases import AliasResolver
from .canonicalizer import EntityCanonicalizer
from .confidence import ConfidenceBreakdown, ConfidenceEngine
from .identifier import IdentifierNormalizer, NormalizedIdentifier
from .ontology import OntologyNormalizer


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Complete normalization result for one biomedical entity."""

    entity: BiomedicalEntity
    identifier: NormalizedIdentifier | None
    confidence: ConfidenceBreakdown


class NormalizationResolver:
    """Coordinate the BioNexus normalization pipeline."""

    def __init__(
        self,
        *,
        alias_resolver: AliasResolver | None = None,
        ontology_normalizer: OntologyNormalizer | None = None,
        canonicalizer: EntityCanonicalizer | None = None,
        confidence_engine: ConfidenceEngine | None = None,
    ) -> None:
        self.logger = get_logger("normalization")
        self.identifier_normalizer = IdentifierNormalizer()
        self.alias_resolver = (
            alias_resolver or AliasResolver()
        )
        self.ontology_normalizer = (
            ontology_normalizer or OntologyNormalizer()
        )
        self.canonicalizer = (
            canonicalizer
            or EntityCanonicalizer(self.alias_resolver)
        )
        self.confidence_engine = (
            confidence_engine or ConfidenceEngine()
        )

    def normalize(
        self,
        entity: BiomedicalEntity,
        correlation_id: str | None = None,
    ) -> NormalizationResult:
        """Normalize one biomedical entity."""
        cid = correlation_id or get_correlation_id()
        if not isinstance(entity, BiomedicalEntity):
            raise TypeError(
                "entity must be a BiomedicalEntity"
            )

        try:
            normalized_identifier = self._normalize_identifier(entity)

            canonical = self.canonicalizer.canonicalize(entity)

            if normalized_identifier is not None and not canonical.canonical_id:
                canonical.canonical_id = normalized_identifier.canonical
                canonical.canonical_source = normalized_identifier.namespace.value

            self._normalize_aliases(canonical)
            self._normalize_ontology_terms(canonical)

            confidence = self.confidence_engine.breakdown(
                canonical
            )

            canonical.confidence = confidence.overall

            return NormalizationResult(
                entity=canonical,
                identifier=normalized_identifier,
                confidence=confidence,
            )
        except Exception as exc:
            self.logger.error("Normalization error for entity %s [%s]: %s", entity.id, cid, exc)
            raise

    def normalize_many(
        self,
        entities: list[BiomedicalEntity],
        correlation_id: str | None = None,
    ) -> list[NormalizationResult]:
        """Normalize multiple biomedical entities."""
        cid = correlation_id or get_correlation_id()
        return [
            self.normalize(entity, correlation_id=cid)
            for entity in entities
        ]

    def resolve(
        self,
        entities: list[BiomedicalEntity],
        correlation_id: str | None = None,
    ) -> list[BiomedicalEntity]:
        """Normalize and merge entities sharing canonical identity."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.debug("[%s] Normalizing and resolving %d entities", cid, len(entities))

        try:
            normalized = [
                result.entity
                for result in self.normalize_many(entities, correlation_id=cid)
            ]

            merged = self.canonicalizer.canonicalize_many(
                normalized
            )

            self.confidence_engine.apply_many(merged)

            return merged
        except Exception as exc:
            self.logger.error("Normalization resolve failed [%s]: %s", cid, exc)
            raise

    def register_aliases(
        self,
        canonical_id: str,
        aliases: list[str],
    ) -> None:
        """Register authoritative aliases for a canonical identifier."""
        self.alias_resolver.register(
            canonical_id,
            aliases,
        )

    def register_ontology_term(
        self,
        term: str,
        *,
        label: str | None = None,
    ) -> None:
        """Register an ontology term in the local ontology registry."""
        self.ontology_normalizer.register_mapping(
            term,
            label=label,
        )

    def _normalize_identifier(
        self,
        entity: BiomedicalEntity,
    ) -> NormalizedIdentifier | None:
        """Normalize the strongest available identifier."""
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
                normalized = self.identifier_normalizer.normalize(
                    candidate
                )
            except (TypeError, ValueError):
                continue

            if normalized.namespace.value == "unknown":
                continue

            return normalized

        return None

    def _normalize_aliases(
        self,
        entity: BiomedicalEntity,
    ) -> None:
        """Resolve locally registered aliases and deduplicate them."""
        aliases = list(entity.aliases)

        if entity.name:
            aliases.append(entity.name)

        if entity.id:
            aliases.append(entity.id)

        resolved_aliases: list[str] = []

        for alias in aliases:
            match = self.alias_resolver.resolve(alias)

            if match is not None:
                resolved_aliases.append(match.canonical_id)

            resolved_aliases.append(alias)

        entity.aliases = self._unique(resolved_aliases)

    def _normalize_ontology_terms(
        self,
        entity: BiomedicalEntity,
    ) -> None:
        """Normalize ontology identifiers without external lookups."""
        normalized_terms: list[str] = []

        for term in entity.ontology_terms:
            try:
                normalized = self.ontology_normalizer.canonicalize(
                    term
                )
            except (TypeError, ValueError):
                # Preserve unrecognized terms rather than silently deleting
                # information from the source entity.
                normalized = term.strip()

            if normalized and normalized not in normalized_terms:
                normalized_terms.append(normalized)

        entity.ontology_terms = normalized_terms

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """Deduplicate strings while preserving order."""
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
    "NormalizationResult",
    "NormalizationResolver",
]
